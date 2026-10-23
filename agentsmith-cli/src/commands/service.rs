//! `agentsmith run --service`. One process, one registry, one MCP poll loop.

use std::collections::{BTreeMap, VecDeque};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use agentsmith_core::config::{Config, ToolsConfig};
use agentsmith_core::redact::redact;
use agentsmith_core::Error;
use agentsmith_llm::{connect, Llm};
use agentsmith_mcp::{McpSupervisor, McpSupervisorConfig};
use agentsmith_runtime::{
    CancellationToken, LoopLimits, LoopMode, Phase, RunEvents, RunId, RunMeta, RunStatus, SpawnHub,
    Supervisor, Transcript,
};
use agentsmith_tools::{register_builtins, BuiltinLimits, SessionPoll, ToolRegistry};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use serde::Serialize;
use tokio::net::TcpListener;

use super::init_tracing;
use super::local_api;
use super::reporter::{self, ReporterHandle};
use super::run::cancel_on_signal;

pub(crate) const MAX_BODY: usize = 1024 * 1024;
const TERMINAL_ROOTS: usize = 32;

pub(crate) struct ServiceInner {
    pub(crate) config: Config,
    pub(crate) llm: Arc<dyn Llm>,
    pub(crate) registry: ToolRegistry,
    pub(crate) sessions: Option<Arc<dyn SessionPoll>>,
    pub(crate) mcp: Option<Arc<McpSupervisor>>,
    pub(crate) supervisor: Arc<Supervisor>,
    pub(crate) events: Arc<dyn RunEvents>,
    pub(crate) hub: Arc<SpawnHub>,
    index: Mutex<Index>,
    pub(crate) cancel: CancellationToken,
    started: Instant,
    shutdown: AtomicBool,
    poll_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    reporter: Mutex<Option<ReporterHandle>>,
    pub(crate) limits: LoopLimits,
    pub(crate) max_output_tokens: u32,
}

#[derive(Default)]
struct Index {
    roots: BTreeMap<RunId, RootRec>,
    terminal_order: VecDeque<RunId>,
    idempotency: BTreeMap<String, (String, RunId)>,
}

struct RootRec {
    mode: LoopMode,
    workdir: PathBuf,
    cancel: CancellationToken,
    terminal: bool,
}

pub(crate) struct Admitted {
    pub(crate) run_id: RunId,
    pub(crate) replay: bool,
}

pub(crate) enum AdmitError {
    Overlap,
    Full,
    Conflict,
}

struct ServiceEvents {
    inner: std::sync::Weak<ServiceInner>,
}

#[async_trait::async_trait]
impl RunEvents for ServiceEvents {
    async fn started(&self, run: RunId, meta: RunMeta) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        inner.supervisor.started(run, meta).await;
    }

    async fn phase(&self, run: RunId, phase: Phase) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        inner.supervisor.phase(run, phase).await;
    }

    async fn iteration(&self, run: RunId, n: u32) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        inner.supervisor.iteration(run, n).await;
    }

    async fn tool_used(&self, run: RunId, name: &str) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        inner.supervisor.tool_used(run, name).await;
    }

    async fn finished(
        &self,
        run: RunId,
        status: RunStatus,
        error_kind: Option<String>,
    ) -> (RunStatus, Option<String>) {
        let Some(inner) = self.inner.upgrade() else {
            return (status, error_kind);
        };
        let result = inner.supervisor.finished(run, status, error_kind).await;
        if result.0.is_terminal() {
            inner.note_terminal(run);
        }
        result
    }

    fn record_terminal(
        &self,
        run: RunId,
        status: RunStatus,
        error_kind: Option<String>,
    ) -> (RunStatus, Option<String>) {
        let Some(inner) = self.inner.upgrade() else {
            return (status, error_kind);
        };
        let result = inner.supervisor.record_terminal(run, status, error_kind);
        if result.0.is_terminal() {
            inner.note_terminal(run);
        }
        result
    }

    fn adjust_live(&self, delta: isize) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        inner.supervisor.add_live(delta);
    }
}

impl ServiceInner {
    pub(crate) fn shutting_down(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    pub(crate) fn authorized(&self, headers: &hyper::HeaderMap) -> bool {
        let expected = self.config.service.token.as_bytes();
        if expected.is_empty() {
            return true;
        }
        let presented = headers
            .get(hyper::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .unwrap_or("");
        tokens_equal(expected, presented.as_bytes())
    }

    pub(crate) fn health(&self) -> HealthBody {
        let active = lock(&self.index)
            .roots
            .values()
            .filter(|root| !root.terminal)
            .count() as u64;
        HealthBody {
            status: "ok",
            uptime_secs: self.started.elapsed().as_secs(),
            active_runs: active,
        }
    }

    pub(crate) fn view(&self, id: RunId) -> Option<RunView> {
        let snap = self.supervisor.snapshot(id)?;
        let mode = lock(&self.index)
            .roots
            .get(&id)
            .map(|root| root.mode)
            .unwrap_or(LoopMode::React);
        let mut children: Vec<_> = self
            .supervisor
            .snapshots()
            .into_iter()
            .filter(|child| child.parent_run_id == Some(id))
            .map(|child| ChildView {
                run_id: child.run_id.to_string(),
                status: child.status,
            })
            .collect();
        children.sort_by(|left, right| left.run_id.cmp(&right.run_id));
        Some(RunView {
            run_id: id.to_string(),
            status: snap.status,
            mode,
            iteration: snap.iteration,
            children,
        })
    }

    /// `true` when `id` is still known. Roots and recorded children cancel.
    /// A live child with no token is not a false success.
    pub(crate) fn cancel_run(&self, id: RunId) -> bool {
        if let Some(root) = lock(&self.index).roots.get(&id) {
            root.cancel.cancel();
            return true;
        }
        if self.hub.cancel_run(id) {
            return true;
        }
        self.supervisor
            .snapshot(id)
            .is_some_and(|snap| snap.status.is_terminal())
    }

    pub(crate) fn spec_hits_down(&self, spec: &agentsmith_runtime::AgentSpec) -> bool {
        let Some(tools) = spec.tools.as_deref() else {
            return false;
        };
        if tools.is_empty() {
            return false;
        }
        let down = self
            .mcp
            .as_ref()
            .map(|mcp| mcp.down_servers())
            .unwrap_or_default();
        tools.iter().any(|name| {
            self.registry.is_mcp_down(name)
                || down.iter().any(|server| {
                    let mut prefix = String::with_capacity(server.len() + 2);
                    prefix.push_str(server);
                    prefix.push_str("__");
                    name.starts_with(&prefix)
                })
        })
    }

    pub(crate) fn root_meta(
        &self,
        mode: LoopMode,
        name: &str,
        spec_id: &str,
        task: &str,
    ) -> RunMeta {
        RunMeta {
            parent_run_id: None,
            depth: 0,
            name: name.to_string(),
            max_iterations: match mode {
                LoopMode::React => self.limits.react_max_iterations,
                LoopMode::Plan => self.limits.step_max_iterations,
            },
            phase: phase_of(mode),
            mode,
            spec_id: spec_id.to_string(),
            task: task.to_string(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn admit(
        &self,
        key: Option<&str>,
        hash: &str,
        workdir: &Path,
        mode: LoopMode,
        cancel: CancellationToken,
        mcp_down: bool,
    ) -> Result<Admitted, AdmitError> {
        let mut index = lock(&self.index);
        if let Some(key) = key {
            if let Some((previous, run_id)) = index.idempotency.get(key) {
                if previous == hash {
                    return Ok(Admitted {
                        run_id: *run_id,
                        replay: true,
                    });
                }
                return Err(AdmitError::Conflict);
            }
        }
        if index
            .roots
            .values()
            .any(|root| !root.terminal && paths_overlap(&root.workdir, workdir))
        {
            return Err(AdmitError::Overlap);
        }
        // A start failure is already terminal, so it does not take a slot.
        if !mcp_down {
            let active = index.roots.values().filter(|root| !root.terminal).count() as u64;
            if active >= self.config.service.max_runs {
                return Err(AdmitError::Full);
            }
        }
        let run_id = agentsmith_core::id::new_id();
        index.roots.insert(
            run_id,
            RootRec {
                mode,
                workdir: workdir.to_path_buf(),
                cancel,
                terminal: mcp_down,
            },
        );
        if mcp_down {
            index.terminal_order.push_back(run_id);
            self.evict(&mut index);
        }
        if let Some(key) = key {
            index
                .idempotency
                .insert(key.to_string(), (hash.to_string(), run_id));
        }
        Ok(Admitted {
            run_id,
            replay: false,
        })
    }

    pub(crate) async fn record_mcp_down(
        &self,
        run_id: RunId,
        mode: LoopMode,
        name: &str,
        spec_id: &str,
        task: &str,
    ) {
        self.events
            .started(run_id, self.root_meta(mode, name, spec_id, task))
            .await;
        self.events
            .finished(run_id, RunStatus::Failed, Some("mcp_down".into()))
            .await;
        let iteration = self
            .supervisor
            .snapshot(run_id)
            .map(|snap| snap.iteration)
            .unwrap_or(0);
        write_terminal(
            &self.config.harness.data_dir,
            run_id,
            RunStatus::Failed,
            Some("mcp_down"),
            iteration,
            phase_of(mode),
        );
    }

    pub(crate) async fn fail_unfinished(&self, run_id: RunId, mode: LoopMode) {
        if self
            .supervisor
            .snapshot(run_id)
            .is_some_and(|snap| snap.status.is_terminal())
        {
            return;
        }
        let iteration = self
            .supervisor
            .snapshot(run_id)
            .map(|snap| snap.iteration)
            .unwrap_or(0);
        self.events
            .finished(run_id, RunStatus::Failed, Some("execution".into()))
            .await;
        write_terminal(
            &self.config.harness.data_dir,
            run_id,
            RunStatus::Failed,
            Some("execution"),
            iteration,
            phase_of(mode),
        );
    }

    fn note_terminal(&self, id: RunId) {
        let mut index = lock(&self.index);
        let Some(root) = index.roots.get_mut(&id) else {
            return;
        };
        if root.terminal {
            return;
        }
        root.terminal = true;
        index.terminal_order.push_back(id);
        self.evict(&mut index);
    }

    fn evict(&self, index: &mut Index) {
        while index.terminal_order.len() > TERMINAL_ROOTS {
            let Some(old) = index.terminal_order.pop_front() else {
                break;
            };
            index.roots.remove(&old);
            index.idempotency.retain(|_, (_, run_id)| *run_id != old);
            let removed = self.supervisor.remove_terminal_tree(old);
            self.hub.forget_cancels(&removed);
        }
    }

    fn cancel_all(&self) {
        for root in lock(&self.index).roots.values() {
            root.cancel.cancel();
        }
    }
}

#[derive(Serialize)]
pub(crate) struct HealthBody {
    pub status: &'static str,
    pub uptime_secs: u64,
    pub active_runs: u64,
}

#[derive(Serialize)]
pub(crate) struct RunView {
    pub run_id: String,
    pub status: RunStatus,
    pub mode: LoopMode,
    pub iteration: u32,
    pub children: Vec<ChildView>,
}

#[derive(Serialize)]
pub(crate) struct ChildView {
    pub run_id: String,
    pub status: RunStatus,
}

pub(crate) fn phase_of(mode: LoopMode) -> Phase {
    match mode {
        LoopMode::React => Phase::React,
        LoopMode::Plan => Phase::Plan,
    }
}

pub(crate) fn paths_overlap(left: &Path, right: &Path) -> bool {
    left == right || left.starts_with(right) || right.starts_with(left)
}

fn tokens_equal(expected: &[u8], presented: &[u8]) -> bool {
    // Full-width length. A `u8` fold treats 256 and 0 as the same length.
    let mut diff = expected.len() ^ presented.len();
    let mut index = 0;
    while index < expected.len() || index < presented.len() {
        let left = expected.get(index).copied().unwrap_or(0);
        let right = presented.get(index).copied().unwrap_or(0);
        diff |= usize::from(left ^ right);
        index += 1;
    }
    diff == 0
}

fn write_terminal(
    data_dir: &Path,
    run_id: RunId,
    status: RunStatus,
    kind: Option<&str>,
    iteration: u32,
    phase: Phase,
) {
    // Status is already on the supervisor. A failed append must not undo it.
    match Transcript::with_log(data_dir, run_id) {
        Ok(mut transcript) => {
            if let Err(err) = transcript.finish(status, kind, iteration, phase) {
                tracing::warn!("{}", redact(&err.to_string()));
            }
        }
        Err(err) => tracing::warn!("{}", redact(&err.to_string())),
    }
}

fn register_service_builtins(registry: &ToolRegistry, tools: &ToolsConfig) -> Result<(), Error> {
    // Unset shell stays off. Foreground uses the opposite default.
    let enable_shell = tools.shell.enabled.unwrap_or(false);
    register_builtins(
        registry,
        &BuiltinLimits {
            max_read_bytes: tools.files.max_read_bytes,
            max_write_bytes: tools.files.max_write_bytes,
            max_output_bytes: tools.max_output_bytes,
            shell_timeout: Duration::from_secs(tools.shell.timeout_secs),
            shell_max_timeout: Duration::from_secs(tools.shell.max_timeout_secs),
            enable_shell,
        },
    )
}

pub struct RunningService {
    inner: Arc<ServiceInner>,
    // Tests read the bound port. The process path does not.
    #[cfg_attr(not(test), allow(dead_code))]
    local_addr: SocketAddr,
    join: Option<tokio::task::JoinHandle<()>>,
}

impl RunningService {
    pub async fn start(
        config: Config,
        llm: Arc<dyn Llm>,
        cancel: CancellationToken,
    ) -> Result<Self, Error> {
        init_tracing(config.harness.log_level);
        if config.service.token.is_empty() {
            tracing::warn!("service.token is empty; loopback auth is disabled");
        }
        let limits = LoopLimits::from_config(&config.loop_)?;
        let max_output_tokens = u32::try_from(config.llm.max_output_tokens)
            .map_err(|_| Error::Config("llm.max_output_tokens does not fit in u32".into()))?;
        let registry = ToolRegistry::new();
        register_service_builtins(&registry, &config.tools)?;
        let (mcp, sessions) = if config.mcp.servers.is_empty() {
            (None, None)
        } else {
            let mcp = McpSupervisor::new(McpSupervisorConfig {
                registry: registry.clone(),
                servers: config.mcp.servers.clone(),
                max_output_bytes: config.tools.max_output_bytes,
                shutdown_grace: Duration::from_secs(config.service.shutdown_grace_secs),
                cancel: cancel.clone(),
            });
            let sessions = mcp.clone() as Arc<dyn SessionPoll>;
            (Some(mcp), Some(sessions))
        };
        let bind = parse_bind(&config.service.bind)?;
        let supervisor = Arc::new(Supervisor::new());
        let hub = SpawnHub::new();
        let inner = Arc::new_cyclic(|weak| {
            let events: Arc<dyn RunEvents> = Arc::new(ServiceEvents {
                inner: weak.clone(),
            });
            ServiceInner {
                config,
                llm,
                registry,
                sessions,
                mcp,
                supervisor,
                events,
                hub,
                index: Mutex::new(Index::default()),
                cancel,
                started: Instant::now(),
                shutdown: AtomicBool::new(false),
                poll_task: Mutex::new(None),
                reporter: Mutex::new(None),
                limits,
                max_output_tokens,
            }
        });
        if let Some(sessions) = &inner.sessions {
            match sessions.poll().await {
                Ok(()) => {}
                Err(Error::Cancelled) => {
                    if let Some(mcp) = &inner.mcp {
                        let _ = mcp.shutdown().await;
                    }
                    return Err(Error::Cancelled);
                }
                Err(err) => tracing::warn!("{}", redact(&err.to_string())),
            }
        }
        if let Some(sessions) = inner.sessions.clone() {
            let task = spawn_poll(sessions, inner.cancel.clone());
            *lock(&inner.poll_task) = Some(task);
        }
        let listener = TcpListener::bind(bind).await?;
        let local_addr = listener.local_addr()?;
        if let Some(handle) = reporter::start(reporter::Parts {
            config: inner.config.clone(),
            mode: agentsmith_protocol::ProcessMode::Service,
            supervisor: Arc::clone(&inner.supervisor),
            registry: inner.registry.clone(),
            mcp: inner.mcp.clone(),
            booted: inner.started,
        }) {
            *lock(&inner.reporter) = Some(handle);
        }
        let serve_inner = Arc::clone(&inner);
        let join = tokio::spawn(async move {
            accept_loop(serve_inner, listener).await;
        });
        Ok(Self {
            inner,
            local_addr,
            join: Some(join),
        })
    }

    // Tests open the socket and force 503. The signal path stops in `accept_loop`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn addr(&self) -> SocketAddr {
        self.local_addr
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn begin_shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::Release);
    }

    pub async fn run_until_cancel(mut self) {
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn shutdown(mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        self.inner.cancel.cancel();
        if let Some(join) = self.join.take() {
            let grace = Duration::from_secs(
                self.inner
                    .config
                    .service
                    .shutdown_grace_secs
                    .saturating_add(2),
            );
            let _ = tokio::time::timeout(grace, join).await;
        }
    }
}

impl Drop for RunningService {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        self.inner.cancel.cancel();
    }
}

async fn accept_loop(inner: Arc<ServiceInner>, listener: TcpListener) {
    loop {
        tokio::select! {
            biased;
            _ = inner.cancel.cancelled() => break,
            accepted = listener.accept() => {
                let Ok((stream, _)) = accepted else {
                    continue;
                };
                let io = TokioIo::new(stream);
                let inner = Arc::clone(&inner);
                tokio::spawn(async move {
                    let service = service_fn(move |req| {
                        let inner = Arc::clone(&inner);
                        async move { Ok::<_, hyper::Error>(local_api::handle(inner, req).await) }
                    });
                    if let Err(err) = http1::Builder::new().serve_connection(io, service).await {
                        tracing::debug!("{}", redact(&err.to_string()));
                    }
                });
            }
        }
    }
    inner.shutdown.store(true, Ordering::Release);
    inner.cancel_all();
    let poll_task = lock(&inner.poll_task).take();
    if let Some(task) = poll_task {
        let _ = task.await;
    }
    // One grace budget. The stopping post uses what it needs; MCP gets the rest.
    // A timeout around MCP shutdown drops the close before it can SIGKILL.
    let reporter = lock(&inner.reporter).take();
    if let Some(reporter) = reporter {
        let deadline =
            Instant::now() + Duration::from_secs(inner.config.service.shutdown_grace_secs);
        reporter.shutdown(deadline).await;
        if let Some(mcp) = &inner.mcp {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let _ = mcp.shutdown_within(remaining).await;
        }
    } else if let Some(mcp) = &inner.mcp {
        let _ = mcp.shutdown().await;
    }
}

fn spawn_poll(
    sessions: Arc<dyn SessionPoll>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(1));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // The startup poll already ran. Drop the immediate first tick.
        ticker.tick().await;
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = ticker.tick() => {
                    match sessions.poll().await {
                        Ok(()) => {}
                        Err(Error::Cancelled) => break,
                        Err(Error::Mcp { .. }) => {}
                        Err(err) => tracing::warn!("{}", redact(&err.to_string())),
                    }
                }
            }
        }
    })
}

pub fn execute(config_path: Option<&Path>) -> Result<ExitCode, Error> {
    let config = agentsmith_core::config::load(config_path)?;
    let llm = connect(
        config.llm.provider,
        &config.llm.base_url,
        config.llm.api_key.clone(),
        Duration::from_secs(config.llm.timeout_secs),
        u32::try_from(config.llm.retries).unwrap_or(u32::MAX),
    )?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let cancel = CancellationToken::new();
        tokio::spawn(cancel_on_signal(cancel.clone()));
        match RunningService::start(config, Arc::new(llm), cancel).await {
            Ok(running) => {
                running.run_until_cancel().await;
                Ok(())
            }
            Err(Error::Cancelled) => Ok(()),
            Err(err) => Err(err),
        }
    })?;
    Ok(ExitCode::SUCCESS)
}

fn parse_bind(bind: &str) -> Result<SocketAddr, Error> {
    if let Ok(addr) = bind.parse() {
        return Ok(addr);
    }
    if let Some((host, port)) = bind.rsplit_once(':') {
        if host.eq_ignore_ascii_case("localhost") {
            if let Ok(port) = port.parse::<u16>() {
                return Ok(SocketAddr::from((Ipv4Addr::LOCALHOST, port)));
            }
        }
    }
    Err(Error::Config(format!(
        "service.bind is not a socket address: {bind}"
    )))
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentsmith_core::config::{McpServerConfig, ToolsConfig};

    #[test]
    fn bearer_compare_rejects_length_aliases() {
        let token = vec![1u8; 300];
        assert!(tokens_equal(&token, &token));
        assert!(!tokens_equal(&token, &token[..44]));
        assert!(!tokens_equal(&token, &[]));
        assert!(!tokens_equal(&[0u8; 256], &[]));
        assert!(!tokens_equal(b"secret", b""));
        assert!(!tokens_equal(b"secret", b"secre"));
        assert!(!tokens_equal(b"secret", b"secretEXTRA"));
        assert!(tokens_equal(b"", b""));
    }
    use agentsmith_llm::{Completion, CompletionRequest, Message, StopReason};
    use serde_json::Value;
    use std::sync::atomic::AtomicU64;

    struct GateLlm;

    #[async_trait::async_trait]
    impl Llm for GateLlm {
        async fn complete(&self, request: CompletionRequest) -> Result<Completion, Error> {
            let fast = request.messages.iter().any(|message| match message {
                Message::User { content } => content.contains("FAST"),
                _ => false,
            });
            if fast {
                return Ok(Completion {
                    message: Message::Assistant {
                        content: Some("done".into()),
                        tool_calls: Vec::new(),
                    },
                    stop: StopReason::EndTurn,
                });
            }
            request.cancel.cancelled().await;
            Err(Error::Cancelled)
        }
    }

    fn scratch() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "agentsmith-service-{}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn test_config(dir: &Path, token: &str, max_runs: u64) -> Config {
        let mut cfg = Config::default();
        cfg.harness.data_dir = dir.join("data");
        cfg.harness.workdir = dir.join("work");
        cfg.service.bind = "127.0.0.1:0".into();
        cfg.service.token = token.into();
        cfg.service.max_runs = max_runs;
        cfg.service.shutdown_grace_secs = 1;
        cfg.llm.model = "fake".into();
        cfg.llm.api_key = "k".into();
        std::fs::create_dir_all(&cfg.harness.workdir).unwrap();
        std::fs::create_dir_all(&cfg.harness.data_dir).unwrap();
        cfg
    }

    async fn boot(cfg: Config) -> RunningService {
        RunningService::start(cfg, Arc::new(GateLlm), CancellationToken::new())
            .await
            .unwrap()
    }

    async fn http(
        addr: SocketAddr,
        method: &str,
        path: &str,
        token: Option<&str>,
        extra: &[(&str, &str)],
        body: &str,
    ) -> (u16, String) {
        let method = method.to_string();
        let path = path.to_string();
        let token = token.map(str::to_string);
        let extra: Vec<(String, String)> = extra
            .iter()
            .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
            .collect();
        let body = body.to_string();
        tokio::task::spawn_blocking(move || {
            http_blocking(addr, &method, &path, token.as_deref(), &extra, &body)
        })
        .await
        .unwrap()
    }

    fn http_blocking(
        addr: SocketAddr,
        method: &str,
        path: &str,
        token: Option<&str>,
        extra: &[(String, String)],
        body: &str,
    ) -> (u16, String) {
        use std::io::{Read, Write};
        let mut stream = std::net::TcpStream::connect(addr).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut req = format!(
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Length: {}\r\n",
            body.len()
        );
        if let Some(token) = token {
            req.push_str(&format!("Authorization: Bearer {token}\r\n"));
        }
        for (name, value) in extra {
            req.push_str(&format!("{name}: {value}\r\n"));
        }
        req.push_str("\r\n");
        req.push_str(body);
        stream.write_all(req.as_bytes()).unwrap();
        let mut buf = String::new();
        let _ = stream.read_to_string(&mut buf);
        let code = buf
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse().ok())
            .unwrap_or(0);
        let body = buf.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
        (code, body)
    }

    fn run_id_of(body: &str) -> String {
        serde_json::from_str::<Value>(body).unwrap()["run_id"]
            .as_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn unset_shell_is_off_for_the_service() {
        let registry = ToolRegistry::new();
        let mut tools = ToolsConfig::default();
        tools.shell.enabled = None;
        register_service_builtins(&registry, &tools).unwrap();
        assert!(!registry.names().iter().any(|name| name == "shell"));
        tools.shell.enabled = Some(false);
        let disabled = ToolRegistry::new();
        register_service_builtins(&disabled, &tools).unwrap();
        assert!(!disabled.names().iter().any(|name| name == "shell"));
        tools.shell.enabled = Some(true);
        let forced = ToolRegistry::new();
        register_service_builtins(&forced, &tools).unwrap();
        assert!(forced.names().iter().any(|name| name == "shell"));
    }

    #[test]
    fn sibling_paths_do_not_overlap() {
        let dir = scratch();
        let foo = dir.join("foo");
        let foobar = dir.join("foobar");
        let child = foo.join("child");
        std::fs::create_dir_all(&child).unwrap();
        std::fs::create_dir_all(&foobar).unwrap();
        let foo = foo.canonicalize().unwrap();
        let foobar = foobar.canonicalize().unwrap();
        let child = child.canonicalize().unwrap();
        assert!(paths_overlap(&foo, &foo));
        assert!(paths_overlap(&foo, &child));
        assert!(!paths_overlap(&foo, &foobar));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn canonical_json_hash_sorts_nested_keys() {
        let left = serde_json::json!({"b":1,"a":{"d":1,"c":2}});
        let right = serde_json::json!({"a":{"c":2,"d":1},"b":1});
        assert_eq!(
            crate::commands::local_api::body_hash(&left),
            crate::commands::local_api::body_hash(&right)
        );
        assert_ne!(
            crate::commands::local_api::body_hash(&left),
            crate::commands::local_api::body_hash(&serde_json::json!({"a":1}))
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn health_is_open_and_a_bad_bearer_is_401() {
        let dir = scratch();
        let service = boot(test_config(&dir, "secret", 2)).await;
        let addr = service.addr();
        let (code, body) = http(addr, "GET", "/health", None, &[], "").await;
        assert_eq!(code, 200, "{body}");
        let health: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(health["status"], "ok");
        assert!(health["uptime_secs"].is_u64());
        assert_eq!(health["active_runs"], 0);
        let (code, _) = http(
            addr,
            "POST",
            "/v1/runs",
            None,
            &[],
            r#"{"task":"hello","mode":"react"}"#,
        )
        .await;
        assert_eq!(code, 401);
        let (code, _) = http(
            addr,
            "POST",
            "/v1/runs",
            Some("nope"),
            &[],
            r#"{"task":"hello","mode":"react"}"#,
        )
        .await;
        assert_eq!(code, 401);
        let (code, _) = http(addr, "GET", "/health", None, &[], "").await;
        assert_eq!(code, 200);
        service.shutdown().await;
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn empty_token_allows_an_unauthenticated_post() {
        let dir = scratch();
        let service = boot(test_config(&dir, "", 2)).await;
        let addr = service.addr();
        let (code, body) = http(
            addr,
            "POST",
            "/v1/runs",
            None,
            &[],
            r#"{"task":"hello","mode":"react"}"#,
        )
        .await;
        assert_eq!(code, 201, "{body}");
        assert!(body.contains("run_id"), "{body}");
        service.shutdown().await;
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn idempotency_dedupes_only_the_same_body() {
        let dir = scratch();
        let service = boot(test_config(&dir, "secret", 8)).await;
        let addr = service.addr();
        let one = dir.join("one");
        let two = dir.join("two");
        std::fs::create_dir_all(&one).unwrap();
        std::fs::create_dir_all(&two).unwrap();
        let first_body = format!(
            r#"{{"task":"hold","mode":"react","workdir":"{}"}}"#,
            one.display()
        );
        let second_body = format!(
            r#"{{"task":"hold","mode":"react","workdir":"{}"}}"#,
            two.display()
        );
        let (code, first) = http(addr, "POST", "/v1/runs", Some("secret"), &[], &first_body).await;
        assert_eq!(code, 201, "{first}");
        let (code, second) =
            http(addr, "POST", "/v1/runs", Some("secret"), &[], &second_body).await;
        assert_eq!(code, 201, "{second}");
        assert_ne!(run_id_of(&first), run_id_of(&second));
        let keyed = r#"{"mode":"react","task":"keyed"}"#;
        let (code, created) = http(
            addr,
            "POST",
            "/v1/runs",
            Some("secret"),
            &[("Idempotency-Key", "k1")],
            keyed,
        )
        .await;
        assert_eq!(code, 201, "{created}");
        let (code, replay) = http(
            addr,
            "POST",
            "/v1/runs",
            Some("secret"),
            &[("Idempotency-Key", "k1")],
            r#"{"task":"keyed","mode":"react"}"#,
        )
        .await;
        assert_eq!(code, 200, "{replay}");
        assert_eq!(run_id_of(&created), run_id_of(&replay));
        let (health_code, health) = http(addr, "GET", "/health", None, &[], "").await;
        assert_eq!(health_code, 200, "{health}");
        let before = serde_json::from_str::<Value>(&health).unwrap()["active_runs"]
            .as_u64()
            .unwrap();
        let (code, denied) = http(
            addr,
            "POST",
            "/v1/runs",
            Some("secret"),
            &[("Idempotency-Key", "k1")],
            r#"{"task":"other","mode":"react"}"#,
        )
        .await;
        assert_eq!(code, 409, "{denied}");
        let (_, health) = http(addr, "GET", "/health", None, &[], "").await;
        let after = serde_json::from_str::<Value>(&health).unwrap()["active_runs"]
            .as_u64()
            .unwrap();
        assert_eq!(before, after);
        service.shutdown().await;
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn overlap_conflict_max_runs_and_finished_replay() {
        let dir = scratch();
        let tree = dir.join("tree");
        let child = tree.join("child");
        let sibling = dir.join("sibling");
        let other = dir.join("other");
        std::fs::create_dir_all(&child).unwrap();
        std::fs::create_dir_all(&sibling).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        let mut cfg = test_config(&dir, "secret", 2);
        cfg.harness.workdir = tree.clone();
        let service = boot(cfg).await;
        let addr = service.addr();
        let root_body = format!(
            r#"{{"task":"hold","mode":"react","workdir":"{}"}}"#,
            tree.display()
        );
        let (code, _) = http(addr, "POST", "/v1/runs", Some("secret"), &[], &root_body).await;
        assert_eq!(code, 201);
        let child_body = format!(
            r#"{{"task":"hold","mode":"react","workdir":"{}"}}"#,
            child.display()
        );
        let (code, _) = http(addr, "POST", "/v1/runs", Some("secret"), &[], &child_body).await;
        assert_eq!(code, 409);
        let sibling_body = format!(
            r#"{{"task":"hold","mode":"react","workdir":"{}"}}"#,
            sibling.display()
        );
        let (code, _) = http(addr, "POST", "/v1/runs", Some("secret"), &[], &sibling_body).await;
        assert_eq!(code, 201);
        let other_body = format!(
            r#"{{"task":"another","mode":"react","workdir":"{}"}}"#,
            other.display()
        );
        let (code, _) = http(addr, "POST", "/v1/runs", Some("secret"), &[], &other_body).await;
        assert_eq!(code, 429);
        service.shutdown().await;

        let mut cfg = test_config(&dir, "secret", 1);
        cfg.harness.workdir = tree.clone();
        let service = boot(cfg).await;
        let addr = service.addr();
        let (code, created) = http(
            addr,
            "POST",
            "/v1/runs",
            Some("secret"),
            &[("Idempotency-Key", "done")],
            r#"{"task":"FAST please","mode":"react"}"#,
        )
        .await;
        assert_eq!(code, 201, "{created}");
        let id = run_id_of(&created);
        let mut finished = false;
        for _ in 0..100 {
            let (code, body) = http(
                addr,
                "GET",
                &format!("/v1/runs/{id}"),
                Some("secret"),
                &[],
                "",
            )
            .await;
            if code == 200 && body.contains("\"status\":\"succeeded\"") {
                finished = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(finished, "fast run did not finish");
        let (code, live) = http(
            addr,
            "POST",
            "/v1/runs",
            Some("secret"),
            &[],
            r#"{"task":"hold","mode":"react"}"#,
        )
        .await;
        assert_eq!(code, 201, "{live}");
        let (code, replay) = http(
            addr,
            "POST",
            "/v1/runs",
            Some("secret"),
            &[("Idempotency-Key", "done")],
            r#"{"task":"FAST please","mode":"react"}"#,
        )
        .await;
        assert_eq!(code, 200, "{replay}");
        assert_eq!(run_id_of(&replay), id);
        let third_body = format!(
            r#"{{"task":"third","mode":"react","workdir":"{}"}}"#,
            sibling.display()
        );
        let (code, _) = http(addr, "POST", "/v1/runs", Some("secret"), &[], &third_body).await;
        assert_eq!(code, 429);
        let (_, health) = http(addr, "GET", "/health", None, &[], "").await;
        assert_eq!(
            serde_json::from_str::<Value>(&health).unwrap()["active_runs"],
            1
        );
        service.shutdown().await;
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cancel_unknown_is_404_and_live_is_202() {
        let dir = scratch();
        let service = boot(test_config(&dir, "secret", 2)).await;
        let addr = service.addr();
        let missing = agentsmith_core::id::new_id();
        let (code, _) = http(
            addr,
            "POST",
            &format!("/v1/runs/{missing}/cancel"),
            Some("secret"),
            &[],
            "",
        )
        .await;
        assert_eq!(code, 404);
        let (code, created) = http(
            addr,
            "POST",
            "/v1/runs",
            Some("secret"),
            &[],
            r#"{"task":"hold","mode":"react"}"#,
        )
        .await;
        assert_eq!(code, 201, "{created}");
        let id = run_id_of(&created);
        let (code, body) = http(
            addr,
            "POST",
            &format!("/v1/runs/{id}/cancel"),
            Some("secret"),
            &[],
            "",
        )
        .await;
        assert_eq!(code, 202, "{body}");
        assert!(body.contains('{'));
        service.shutdown().await;
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn shutting_down_returns_503() {
        let dir = scratch();
        let service = boot(test_config(&dir, "secret", 2)).await;
        let addr = service.addr();
        service.begin_shutdown();
        let (code, _) = http(addr, "GET", "/health", None, &[], "").await;
        assert_eq!(code, 503);
        let (code, _) = http(
            addr,
            "POST",
            "/v1/runs",
            Some("secret"),
            &[],
            r#"{"task":"hold","mode":"react"}"#,
        )
        .await;
        assert_eq!(code, 503);
        service.shutdown().await;
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn down_server_fails_the_run_at_start_and_health_stays_up() {
        let dir = scratch();
        let mut cfg = test_config(&dir, "secret", 2);
        let mut servers = BTreeMap::new();
        servers.insert(
            "fs".into(),
            McpServerConfig {
                command: peer_bin(),
                args: vec!["fail-init".into()],
                required: true,
                connect_timeout_secs: 2,
                ..McpServerConfig::default()
            },
        );
        cfg.mcp.servers = servers;
        let spec = dir.join("agent.toml");
        std::fs::write(
            &spec,
            "id = \"mcp\"\nname = \"mcp\"\nmode = \"react\"\nsystem_prompt = \"x\"\ntools = [\"fs__echo\"]\nmodel = \"fake\"\n",
        )
        .unwrap();
        let service = boot(cfg).await;
        let addr = service.addr();
        let body = format!(
            r#"{{"task":"use the tool","spec_path":"{}"}}"#,
            spec.display()
        );
        let (code, created) = http(addr, "POST", "/v1/runs", Some("secret"), &[], &body).await;
        assert_eq!(code, 201, "{created}");
        let id = run_id_of(&created);
        let (code, view) = http(
            addr,
            "GET",
            &format!("/v1/runs/{id}"),
            Some("secret"),
            &[],
            "",
        )
        .await;
        assert_eq!(code, 200, "{view}");
        let view: Value = serde_json::from_str(&view).unwrap();
        assert_eq!(view["status"], "failed");
        assert_eq!(view["mode"], "react");
        assert!(view.get("children").unwrap().is_array());
        assert!(view.get("task").is_none());
        let jsonl =
            std::fs::read_to_string(dir.join("data").join("runs").join(format!("{id}.jsonl")))
                .unwrap();
        assert!(jsonl.contains("\"error_kind\":\"mcp_down\""), "{jsonl}");
        let (code, health) = http(addr, "GET", "/health", None, &[], "").await;
        assert_eq!(code, 200, "{health}");
        service.shutdown().await;
        let _ = std::fs::remove_dir_all(dir);
    }

    fn peer_bin() -> String {
        if let Some(path) = option_env!("CARGO_BIN_EXE_mcp_stdio_peer") {
            return path.to_string();
        }
        let exe = std::env::current_exe().expect("test executable");
        exe.parent()
            .and_then(|dir| dir.parent())
            .expect("target profile dir")
            .join("mcp-stdio-peer")
            .display()
            .to_string()
    }
}
