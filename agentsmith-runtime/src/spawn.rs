//! One join-all implementation. The plan walker and `spawn_subagents` both call it.
//!
//! `SpawnRequest.child_depth` is the depth to store. This module does not add 1.
//! Only the tool passes `caller.depth + 1`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Duration;

use agentsmith_core::config::FailurePolicy;
use agentsmith_core::redact::redact;
use agentsmith_core::{Error, ToolErrorKind};
use agentsmith_llm::Llm;
use agentsmith_tools::{SessionPoll, Tool, ToolCtx, ToolDefinition, ToolOutput, ToolRegistry};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::loop_::{RunContext, RunEvents, RunId, RunMeta, RunOutcome, RunStatus};
use crate::react::ReactLoop;
use crate::run_tools::{RunKind, RunTools, ToolFilter, SPAWN_SUBAGENTS};
use crate::spec::AgentSpec;
use crate::transcript::Transcript;
use crate::CancellationToken;

const CANCEL_GRACE: Duration = Duration::from_secs(2);
const SUMMARY_MAX: usize = 4 * 1024;
const ERROR_MAX: usize = 512;
const TIMEOUT_MAX_SECS: u64 = 600;

/// Who asked for the join. The depth on the request is already the child's depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChildOrigin {
    /// Plan step. Same depth as the walker. No subagent suffix.
    Step,
    /// Model `spawn_subagents` call. The tool already added 1 to the depth.
    Spawned,
}

pub struct SpawnRequest {
    pub parent: RunId,
    /// Depth to assign. Not incremented here.
    pub child_depth: u32,
    pub agents: Vec<ChildSpec>,
    pub timeout: Duration,
    pub failure_policy: FailurePolicy,
    pub cancel: CancellationToken,
    pub origin: ChildOrigin,
}

pub struct ChildSpec {
    pub name: String,
    pub task: String,
    pub context: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ChildOutcome {
    pub name: String,
    pub run_id: RunId,
    pub status: RunStatus,
    pub summary: String,
    pub error_kind: Option<String>,
    pub error_message: Option<String>,
}

pub struct JoinOutcome {
    pub agents: Vec<ChildOutcome>,
}

struct CallerInfo {
    depth: u32,
    system_prompt: String,
}

/// Drops the caller registration when the React loop returns.
pub struct CallerGuard {
    spawner: Arc<SubagentSpawner>,
    run_id: RunId,
}

impl Drop for CallerGuard {
    fn drop(&mut self) {
        lock(&self.spawner.callers).remove(&self.run_id);
    }
}

struct LiveSlots {
    current: AtomicUsize,
    max: usize,
}

impl LiveSlots {
    fn new(max: usize) -> Self {
        Self {
            current: AtomicUsize::new(0),
            max,
        }
    }

    fn get(&self) -> usize {
        self.current.load(Ordering::Acquire)
    }

    fn try_acquire(&self, n: usize) -> bool {
        let mut current = self.current.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(n) else {
                return false;
            };
            if next > self.max {
                return false;
            }
            match self.current.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }

    fn release(&self, n: usize) {
        let prev = self.current.fetch_sub(n, Ordering::AcqRel);
        debug_assert!(prev >= n, "live_subagents released twice");
    }
}

pub struct SpawnerConfig {
    pub registry: ToolRegistry,
    pub llm: Arc<dyn Llm>,
    pub model: String,
    pub max_output_tokens: u32,
    pub temperature: Option<f32>,
    pub max_output_bytes: u64,
    pub workdir: PathBuf,
    pub data_dir: PathBuf,
    pub spec: AgentSpec,
    pub allow: Option<Vec<String>>,
    pub limits: crate::loop_::LoopLimits,
    pub events: Arc<dyn RunEvents>,
    pub sessions: Option<Arc<dyn SessionPoll>>,
}

pub struct SubagentSpawner {
    registry: ToolRegistry,
    llm: Arc<dyn Llm>,
    model: String,
    max_output_tokens: u32,
    temperature: Option<f32>,
    max_output_bytes: u64,
    workdir: PathBuf,
    data_dir: PathBuf,
    spec: AgentSpec,
    allow: Option<Vec<String>>,
    limits: crate::loop_::LoopLimits,
    events: Arc<dyn RunEvents>,
    sessions: Option<Arc<dyn SessionPoll>>,
    live: Arc<LiveSlots>,
    callers: Mutex<std::collections::BTreeMap<RunId, CallerInfo>>,
}

impl SubagentSpawner {
    pub fn new(config: SpawnerConfig) -> Arc<Self> {
        Arc::new(Self {
            live: Arc::new(LiveSlots::new(config.limits.max_live_subagents)),
            registry: config.registry,
            llm: config.llm,
            model: config.model,
            max_output_tokens: config.max_output_tokens,
            temperature: config.temperature,
            max_output_bytes: config.max_output_bytes,
            workdir: config.workdir,
            data_dir: config.data_dir,
            spec: config.spec,
            allow: config.allow,
            limits: config.limits,
            events: config.events,
            sessions: config.sessions,
            callers: Mutex::new(std::collections::BTreeMap::new()),
        })
    }

    /// Register `spawn_subagents` on the process map. The tool holds a `Weak`
    /// so the registry and the spawner do not keep each other alive.
    pub fn install(self: &Arc<Self>) -> Result<(), Error> {
        self.registry
            .register(Arc::new(SpawnSubagents::new(Arc::downgrade(self))))
    }

    pub fn live(&self) -> usize {
        self.live.get()
    }

    pub fn enter(
        self: &Arc<Self>,
        run_id: RunId,
        depth: u32,
        system_prompt: String,
    ) -> CallerGuard {
        lock(&self.callers).insert(
            run_id,
            CallerInfo {
                depth,
                system_prompt,
            },
        );
        CallerGuard {
            spawner: Arc::clone(self),
            run_id,
        }
    }

    fn caller(&self, run_id: RunId) -> Option<CallerInfo> {
        lock(&self.callers).get(&run_id).map(|info| CallerInfo {
            depth: info.depth,
            system_prompt: info.system_prompt.clone(),
        })
    }

    pub async fn spawn_and_join(
        self: &Arc<Self>,
        request: SpawnRequest,
    ) -> Result<JoinOutcome, Error> {
        let n = request.agents.len();
        if n == 0 || n > self.limits.max_fanout {
            return Err(limit_error(
                "spawn_subagents fan-out is outside 1..=max_fanout",
            ));
        }
        // Already cancelled: do not take a slot and do not start a child.
        if request.cancel.is_cancelled() {
            return Err(cancelled_error());
        }
        let kind = match request.origin {
            ChildOrigin::Step => RunKind::Step,
            ChildOrigin::Spawned => RunKind::Spawned,
        };
        let subagent_tools = match request.origin {
            ChildOrigin::Spawned => self.spec.subagent_tools.as_deref(),
            ChildOrigin::Step => None,
        };
        // Fail closed before taking slots. A bad subagent_tools list starts nothing.
        let tools = RunTools::resolve_for(
            self.registry.clone(),
            ToolFilter {
                allow: self.allow.as_deref(),
                spec_tools: self.spec.tools.as_deref(),
                subagent_tools,
                kind,
                depth: request.child_depth,
                max_subagent_depth: self.limits.max_subagent_depth,
            },
        )?;
        if request.cancel.is_cancelled() {
            return Err(cancelled_error());
        }
        if !self.live.try_acquire(n) {
            return Err(limit_error("live_subagents limit"));
        }
        // Slots not yet handed to a child are released if we return early.
        let mut hold = SlotHold {
            live: Arc::clone(&self.live),
            left: n,
        };

        let parent_prompt = self
            .caller(request.parent)
            .map(|info| info.system_prompt)
            .unwrap_or_else(|| self.spec.system_prompt.clone());

        let (tx, mut rx) = mpsc::channel(n);
        let mut children = Vec::with_capacity(n);
        for (index, agent) in request.agents.into_iter().enumerate() {
            if request.cancel.is_cancelled() {
                break;
            }
            let run_id = agentsmith_core::id::new_id();
            let cancel = request.cancel.child_token();
            let spec = child_spec(
                &self.spec,
                &parent_prompt,
                &agent.name,
                request.child_depth,
                request.origin,
            );
            self.events
                .started(
                    run_id,
                    RunMeta {
                        parent_run_id: Some(request.parent),
                        depth: request.child_depth,
                        name: agent.name.clone(),
                        max_iterations: self.limits.step_max_iterations,
                        phase: crate::loop_::Phase::React,
                    },
                )
                .await;
            // `started` awaits, so cancel can land after admission and before the task exists.
            if request.cancel.is_cancelled() {
                self.events
                    .finished(run_id, RunStatus::Cancelled, None)
                    .await;
                break;
            }
            let ctx = RunContext {
                run_id,
                parent_run_id: Some(request.parent),
                depth: request.child_depth,
                name: agent.name.clone(),
                spec,
                task: agent.task,
                extra_context: agent.context,
                model: self.model.clone(),
                max_output_tokens: self.max_output_tokens,
                temperature: self.temperature,
                max_output_bytes: self.max_output_bytes,
                max_iterations: self.limits.step_max_iterations,
                workdir: self.workdir.clone(),
                llm: Arc::clone(&self.llm),
                tools: tools.clone(),
                limits: self.limits.clone(),
                cancel: cancel.clone(),
                events: Arc::clone(&self.events),
                transcript: Transcript::empty(),
                spawner: Some(Arc::clone(self)),
                sessions: self.sessions.clone(),
            };
            let data_dir = self.data_dir.clone();
            let events = Arc::clone(&self.events);
            let sender = tx.clone();
            let name = ctx.name.clone();
            let handle = tokio::spawn(async move {
                let mut report = Report {
                    tx: sender,
                    index,
                    outcome: failed_outcome(&name, run_id, "panic", "subagent task panicked"),
                };
                let outcome = run_child(ctx, &data_dir, &events).await;
                report.outcome = outcome;
            });
            children.push(Inflight {
                name: agent.name,
                run_id,
                cancel,
                handle: Some(handle),
                outcome: None,
                released: false,
                live: Arc::clone(&self.live),
            });
            hold.left -= 1;
        }
        drop(tx);
        // Slots never handed to a child. Spawned children release their own.
        drop(hold);

        if children.is_empty() {
            return Err(cancelled_error());
        }

        self.await_children(
            &mut children,
            &mut rx,
            request.timeout,
            request.failure_policy,
            &request.cancel,
        )
        .await;

        let agents = children
            .into_iter()
            .map(|mut child| {
                child.release();
                child.outcome.take().unwrap_or_else(|| {
                    failed_outcome(
                        &child.name,
                        child.run_id,
                        "execution",
                        "child produced no outcome",
                    )
                })
            })
            .collect();
        Ok(JoinOutcome { agents })
    }

    async fn await_children(
        &self,
        children: &mut [Inflight],
        rx: &mut mpsc::Receiver<(usize, ChildOutcome)>,
        timeout: Duration,
        policy: FailurePolicy,
        parent: &CancellationToken,
    ) {
        let timeout_at = tokio::time::Instant::now() + timeout;
        let mut drain: Option<Drain> = None;
        // n ≤ max_fanout (config ceiling 8). A larger fan-out never reaches this wait.
        loop {
            if children.iter().all(|child| child.outcome.is_some()) {
                break;
            }
            let deadline = drain
                .as_ref()
                .map(|item| item.deadline)
                .unwrap_or(timeout_at);
            tokio::select! {
                biased;
                incoming = rx.recv() => {
                    match incoming {
                        Some((index, outcome)) => {
                            self.observe(children, index, outcome, policy, &mut drain).await;
                        }
                        None => {
                            self.detach_remaining(children, DetachKind::Failed).await;
                            break;
                        }
                    }
                }
                _ = parent.cancelled(), if !matches!(drain.as_ref(), Some(Drain { kind: DetachKind::Parent, .. })) => {
                    begin_drain(children, &mut drain, DetachKind::Parent);
                }
                _ = tokio::time::sleep_until(deadline) => {
                    if drain.is_none() {
                        begin_drain(children, &mut drain, DetachKind::Timeout);
                    } else {
                        let kind = drain.as_ref().map(|item| item.kind).unwrap_or(DetachKind::Timeout);
                        self.detach_remaining(children, kind).await;
                        break;
                    }
                }
            }
        }
    }

    async fn observe(
        &self,
        children: &mut [Inflight],
        index: usize,
        outcome: ChildOutcome,
        policy: FailurePolicy,
        drain: &mut Option<Drain>,
    ) {
        if children.get(index).is_none() {
            return;
        }
        if children[index].outcome.is_some() {
            return;
        }
        let fail_fast = policy == FailurePolicy::FailFast && outcome.status == RunStatus::Failed;
        children[index].release();
        children[index].handle.take();
        let run_id = children[index].run_id;
        let status = outcome.status;
        let error_kind = outcome.error_kind.clone();
        children[index].outcome = Some(outcome);
        // The child loop usually records this first. A panic never does.
        // A second call does not overwrite a terminal status.
        self.events.finished(run_id, status, error_kind).await;
        if fail_fast && drain.is_none() {
            begin_drain(children, drain, DetachKind::FailFast);
        }
    }

    async fn detach_remaining(&self, children: &mut [Inflight], kind: DetachKind) {
        for child in children.iter_mut() {
            if child.outcome.is_some() {
                continue;
            }
            let outcome = match kind {
                DetachKind::Timeout => ChildOutcome {
                    name: child.name.clone(),
                    run_id: child.run_id,
                    status: RunStatus::TimedOut,
                    summary: String::new(),
                    error_kind: Some("timeout".to_string()),
                    error_message: Some(redacted_message("timed out")),
                },
                // fail_fast and parent cancel: grace stragglers are cancelled, not timed_out.
                DetachKind::FailFast | DetachKind::Parent => ChildOutcome {
                    name: child.name.clone(),
                    run_id: child.run_id,
                    status: RunStatus::Cancelled,
                    summary: String::new(),
                    error_kind: None,
                    error_message: Some(redacted_message("cancelled")),
                },
                DetachKind::Failed => failed_outcome(
                    &child.name,
                    child.run_id,
                    "execution",
                    "child ended without a result",
                ),
            };
            // Release once, then drop the handle so the task detaches.
            child.release();
            let handle = child.handle.take();
            drop(handle);
            self.events
                .finished(child.run_id, outcome.status, outcome.error_kind.clone())
                .await;
            child.outcome = Some(outcome);
        }
    }
}

fn child_spec(
    parent: &AgentSpec,
    parent_prompt: &str,
    name: &str,
    depth: u32,
    origin: ChildOrigin,
) -> AgentSpec {
    let mut spec = parent.clone();
    if origin == ChildOrigin::Spawned {
        spec.system_prompt = subagent_prompt(parent_prompt, name, depth);
    }
    spec
}

fn subagent_prompt(parent: &str, name: &str, depth: u32) -> String {
    format!(
        "{parent}\n\nYou are subagent {name} at depth {depth}. Do only the assigned task. The context field is all you know about the parent. Do not assume you can see the parent transcript."
    )
}

async fn run_child(
    mut ctx: RunContext,
    data_dir: &std::path::Path,
    events: &Arc<dyn RunEvents>,
) -> ChildOutcome {
    let name = ctx.name.clone();
    let run_id = ctx.run_id;
    match Transcript::with_log(data_dir, run_id) {
        Ok(transcript) => ctx.transcript = transcript,
        Err(err) => {
            let outcome = failed_outcome(&name, run_id, "execution", &err.to_string());
            events
                .finished(run_id, outcome.status, outcome.error_kind.clone())
                .await;
            return outcome;
        }
    }
    match ReactLoop.run(ctx).await {
        Ok(outcome) => child_from_run(name, run_id, outcome),
        Err(err) => {
            let outcome = failed_outcome(&name, run_id, "execution", &err.to_string());
            events
                .finished(run_id, outcome.status, outcome.error_kind.clone())
                .await;
            outcome
        }
    }
}

fn child_from_run(name: String, run_id: RunId, outcome: RunOutcome) -> ChildOutcome {
    let assistant = cut_bytes(&last_assistant(&outcome.transcript), SUMMARY_MAX);
    let failed = outcome.status != RunStatus::Succeeded;
    ChildOutcome {
        name,
        run_id,
        status: outcome.status,
        summary: if failed { String::new() } else { assistant },
        error_kind: outcome.error_kind,
        error_message: if failed {
            Some(redacted_message(&outcome.summary))
        } else {
            None
        },
    }
}

fn last_assistant(transcript: &[agentsmith_llm::Message]) -> String {
    transcript
        .iter()
        .rev()
        .find_map(|message| match message {
            agentsmith_llm::Message::Assistant { content, .. } => content.clone(),
            _ => None,
        })
        .unwrap_or_default()
}

fn failed_outcome(name: &str, run_id: RunId, kind: &str, message: &str) -> ChildOutcome {
    ChildOutcome {
        name: name.to_string(),
        run_id,
        status: RunStatus::Failed,
        summary: String::new(),
        error_kind: Some(kind.to_string()),
        error_message: Some(redacted_message(message)),
    }
}

fn redacted_message(message: &str) -> String {
    cut_bytes(&redact(message), ERROR_MAX)
}

/// Byte cap on a char boundary. `str::floor_char_boundary` is newer than MSRV 1.88.
pub(crate) fn cut_bytes(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut end = max_bytes.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

pub fn join_json(outcome: &JoinOutcome) -> String {
    let ok = !outcome.agents.is_empty()
        && outcome
            .agents
            .iter()
            .all(|agent| agent.status == RunStatus::Succeeded);
    let agents: Vec<Value> = outcome.agents.iter().map(agent_json).collect();
    json!({
        "ok": ok,
        "joined": "all",
        "agents": agents,
    })
    .to_string()
}

fn agent_json(agent: &ChildOutcome) -> Value {
    let mut value = serde_json::Map::new();
    value.insert("name".into(), json!(agent.name));
    value.insert("run_id".into(), json!(agent.run_id.to_string()));
    value.insert("status".into(), json!(agent.status));
    if agent.status == RunStatus::Succeeded {
        value.insert("summary".into(), json!(agent.summary));
    } else {
        let kind = agent
            .error_kind
            .clone()
            .unwrap_or_else(|| default_kind(agent.status).to_string());
        let message = agent.error_message.clone().unwrap_or_default();
        value.insert(
            "error".into(),
            json!({
                "kind": kind,
                "message": message,
            }),
        );
    }
    Value::Object(value)
}

fn default_kind(status: RunStatus) -> &'static str {
    match status {
        RunStatus::Cancelled => "cancelled",
        RunStatus::TimedOut => "timeout",
        RunStatus::MaxIterations => "max_iterations",
        RunStatus::Partial => "partial",
        _ => "execution",
    }
}

struct SlotHold {
    live: Arc<LiveSlots>,
    left: usize,
}

impl Drop for SlotHold {
    fn drop(&mut self) {
        if self.left > 0 {
            self.live.release(self.left);
        }
    }
}

struct Inflight {
    name: String,
    run_id: RunId,
    cancel: CancellationToken,
    handle: Option<JoinHandle<()>>,
    outcome: Option<ChildOutcome>,
    released: bool,
    live: Arc<LiveSlots>,
}

impl Inflight {
    fn release(&mut self) {
        // Detach and a later task exit both land here. The flag makes the second a no-op.
        if self.released {
            return;
        }
        self.released = true;
        self.live.release(1);
    }
}

impl Drop for Inflight {
    fn drop(&mut self) {
        self.release();
    }
}

struct Report {
    tx: mpsc::Sender<(usize, ChildOutcome)>,
    index: usize,
    outcome: ChildOutcome,
}

impl Drop for Report {
    fn drop(&mut self) {
        let _ = self.tx.try_send((self.index, self.outcome.clone()));
    }
}

#[derive(Clone, Copy)]
enum DetachKind {
    Timeout,
    FailFast,
    Parent,
    Failed,
}

struct Drain {
    kind: DetachKind,
    deadline: tokio::time::Instant,
}

fn begin_drain(children: &mut [Inflight], drain: &mut Option<Drain>, kind: DetachKind) {
    for child in children.iter() {
        if child.outcome.is_none() {
            child.cancel.cancel();
        }
    }
    match drain {
        None => {
            *drain = Some(Drain {
                kind,
                deadline: tokio::time::Instant::now() + CANCEL_GRACE,
            });
        }
        Some(existing) => {
            if matches!(kind, DetachKind::Parent) {
                existing.kind = DetachKind::Parent;
            }
        }
    }
}

fn limit_error(message: &str) -> Error {
    Error::Tool {
        tool: SPAWN_SUBAGENTS.to_string(),
        kind: ToolErrorKind::Limit,
        message: message.to_string(),
    }
}

fn cancelled_error() -> Error {
    Error::Tool {
        tool: SPAWN_SUBAGENTS.to_string(),
        kind: ToolErrorKind::Cancelled,
        message: "cancelled".into(),
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}

struct SpawnSubagents {
    spawner: Weak<SubagentSpawner>,
}

impl SpawnSubagents {
    fn new(spawner: Weak<SubagentSpawner>) -> Self {
        Self { spawner }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpawnArgs {
    agents: Vec<AgentArg>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    failure_policy: Option<FailurePolicy>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentArg {
    name: String,
    task: String,
    #[serde(default)]
    context: Option<String>,
}

#[async_trait::async_trait]
impl Tool for SpawnSubagents {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: SPAWN_SUBAGENTS.to_string(),
            description: "Run subagents concurrently and return only after all of them finish, fail, or time out. Their transcripts are not returned.".to_string(),
            parameters: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["agents"],
                "properties": {
                    "agents": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["name", "task"],
                            "properties": {
                                "name": { "type": "string", "maxLength": 64 },
                                "task": { "type": "string", "maxLength": 8000 },
                                "context": { "type": "string", "maxLength": 8000 }
                            }
                        }
                    },
                    "timeout_secs": { "type": "integer", "minimum": 1, "maximum": 600 },
                    "failure_policy": { "type": "string", "enum": ["continue", "fail_fast"] }
                }
            }),
        }
    }

    async fn call(&self, ctx: ToolCtx, arguments: Value) -> Result<ToolOutput, Error> {
        // Do not take the registry lock. The join awaits child runs that call tools.
        let Some(spawner) = self.spawner.upgrade() else {
            return Err(tool_error(
                ToolErrorKind::Execution,
                "spawn_subagents is unavailable",
            ));
        };
        let args: SpawnArgs = match serde_json::from_value(arguments) {
            Ok(args) => args,
            Err(err) => {
                return Err(tool_error(
                    ToolErrorKind::InvalidArgs,
                    format!("invalid arguments: {err}"),
                ))
            }
        };
        if let Some(message) = validate_spawn_args(&args) {
            return Err(tool_error(ToolErrorKind::InvalidArgs, message));
        }
        let n = args.agents.len();
        if n == 0 || n > spawner.limits.max_fanout {
            return Err(limit_error(
                "spawn_subagents fan-out is outside 1..=max_fanout",
            ));
        }
        let Some(caller) = spawner.caller(ctx.run_id) else {
            return Err(tool_error(
                ToolErrorKind::Execution,
                "spawn_subagents has no caller",
            ));
        };
        // Tool path only. The walker passes its own depth and does not land here.
        let child_depth = match caller.depth.checked_add(1) {
            Some(depth) => depth,
            None => return Err(limit_error("max_subagent_depth")),
        };
        if child_depth > spawner.limits.max_subagent_depth {
            return Err(limit_error("max_subagent_depth"));
        }
        let timeout = match args.timeout_secs {
            Some(secs) => Duration::from_secs(secs),
            None => spawner.limits.spawn_timeout,
        };
        let failure_policy = args.failure_policy.unwrap_or(spawner.limits.failure_policy);
        let agents = args
            .agents
            .into_iter()
            .map(|agent| ChildSpec {
                name: agent.name,
                task: agent.task,
                context: agent.context,
            })
            .collect();
        let outcome = spawner
            .spawn_and_join(SpawnRequest {
                parent: ctx.run_id,
                child_depth,
                agents,
                timeout,
                failure_policy,
                cancel: ctx.cancel,
                origin: ChildOrigin::Spawned,
            })
            .await?;
        Ok(ToolOutput {
            content: join_json(&outcome),
        })
    }
}

fn validate_spawn_args(args: &SpawnArgs) -> Option<String> {
    if let Some(secs) = args.timeout_secs {
        if !(1..=TIMEOUT_MAX_SECS).contains(&secs) {
            return Some("timeout_secs must be from 1 to 600".into());
        }
    }
    for agent in &args.agents {
        if agent.name.chars().count() > 64 {
            return Some("agent name is longer than 64 characters".into());
        }
        if agent.task.chars().count() > 8000 {
            return Some("agent task is longer than 8000 characters".into());
        }
        if agent
            .context
            .as_ref()
            .is_some_and(|context| context.chars().count() > 8000)
        {
            return Some("agent context is longer than 8000 characters".into());
        }
    }
    None
}

fn tool_error(kind: ToolErrorKind, message: impl Into<String>) -> Error {
    Error::Tool {
        tool: SPAWN_SUBAGENTS.to_string(),
        kind,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cut_bytes_stops_on_a_char_boundary() {
        let text = "ééééé";
        assert_eq!(cut_bytes(text, 3), "é");
        assert_eq!(cut_bytes(text, 2), "é");
        assert_eq!(cut_bytes(text, 1), "");
        assert_eq!(cut_bytes("ok", 8), "ok");
        let cut = cut_bytes(text, 512);
        assert!(cut.is_char_boundary(cut.len()));
    }
}
