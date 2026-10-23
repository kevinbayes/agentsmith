//! Push-only AHSP 1.0 snapshots. An ack cannot start, stop, or kill anything.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use agentsmith_core::config::Config;
use agentsmith_core::redact::{redact, take_chars};
use agentsmith_mcp::{McpSnapshot, McpSnapshotStatus, McpSupervisor};
use agentsmith_protocol as ahsp;
use agentsmith_runtime::{LoopMode, Phase, RunSnapshot, RunStatus, Supervisor};
use agentsmith_tools::ToolRegistry;
use chrono::{DateTime, Utc};
use serde_json::Value;
use tokio::sync::{mpsc, Notify};
use tokio::task::JoinHandle;
use uuid::Uuid;

use super::host_watch;

const USER_AGENT: &str = concat!("agentsmith/", env!("CARGO_PKG_VERSION"), " ahsp/1.0");
const MAX_ROOTS: usize = 8;
const MAX_RUNS: usize = 64;
const MAX_CHILDREN: usize = 16;
const MAX_TOOLS: usize = 128;
const MAX_SERVERS: usize = 32;
const MAX_TOOL_USE: usize = 64;
const SKELETON: u8 = 7;
const LOG_CHARS: usize = 512;
const CLOCK_SKEW_SECS: i64 = 5 * 60;

static COMMANDS_WARNED: AtomicBool = AtomicBool::new(false);
static CLOCK_WARNED: AtomicBool = AtomicBool::new(false);
static COMMAND_WARNINGS: AtomicUsize = AtomicUsize::new(0);

pub(crate) struct Parts {
    pub config: Config,
    pub mode: ahsp::ProcessMode,
    pub supervisor: Arc<Supervisor>,
    pub registry: ToolRegistry,
    pub mcp: Option<Arc<McpSupervisor>>,
    pub booted: Instant,
}

pub(crate) struct ReporterHandle {
    stop: mpsc::Sender<Instant>,
    join: JoinHandle<()>,
}

impl ReporterHandle {
    /// Send the stopping snapshot if grace remains. MCP shutdown stays with the caller.
    pub async fn shutdown(self, deadline: Instant) {
        let _ = self.stop.try_send(deadline);
        let remaining = deadline.saturating_duration_since(Instant::now());
        let mut join = self.join;
        tokio::select! {
            _ = &mut join => {}
            _ = tokio::time::sleep(remaining) => {
                join.abort();
            }
        }
    }
}

struct CachedToken {
    access: String,
    expires_at: Instant,
}

struct Reporter {
    config: Config,
    mode: ahsp::ProcessMode,
    supervisor: Arc<Supervisor>,
    registry: ToolRegistry,
    mcp: Option<Arc<McpSupervisor>>,
    booted: Instant,
    instance_id: Uuid,
    fingerprint: String,
    client: reqwest::Client,
    timeout: Duration,
    skew: Duration,
    budget: u64,
    interval_secs: u64,
    sequence: u64,
    cached: Option<CachedToken>,
    secrets: Vec<String>,
    disabled: bool,
    last_post: Option<Instant>,
}

#[derive(Clone)]
struct RunSlot {
    id: Uuid,
    root: bool,
    terminal: bool,
    started: SystemTime,
    finished: Option<SystemTime>,
}

struct Draft {
    report: ahsp::StateReport,
    order: Vec<RunSlot>,
}

struct Chosen {
    snap: RunSnapshot,
    children: Vec<ahsp::ChildRef>,
}

struct Prepared {
    body: String,
    stage: u8,
    report_id: Uuid,
}

enum AckOutcome {
    Success { next_interval: Option<u64> },
    Rejected,
}

enum Posted {
    Ok(String),
    Auth,
    TooLarge,
    Unsupported,
    Retry(Duration),
    GiveUp { status: u16, body: String },
}

/// Enabled only when endpoint, token URL, client id, and client secret are set.
/// A true `enabled` with any of those empty logs once and stays off.
pub(crate) fn activate(config: &Config) -> bool {
    if !config.reporter.enabled {
        return false;
    }
    let incomplete = config.reporter.endpoint.is_empty()
        || config.reporter.token_url.is_empty()
        || config.reporter.client_id.is_empty()
        || config.reporter.client_secret.is_empty();
    if incomplete {
        tracing::warn!(
            "reporter.enabled is set but endpoint, token_url, client_id, or client_secret is empty; reporter stays disabled"
        );
        return false;
    }
    true
}

pub(crate) fn start(parts: Parts) -> Option<ReporterHandle> {
    let reporter = Reporter::try_new(parts)?;
    let notify = reporter.supervisor.root_terminal();
    let (stop, rx) = mpsc::channel(1);
    let join = tokio::spawn(run_loop(reporter, notify, rx));
    Some(ReporterHandle { stop, join })
}

async fn run_loop(mut reporter: Reporter, notify: Arc<Notify>, mut stop: mpsc::Receiver<Instant>) {
    // The first tick is the ready snapshot. Later ticks start after the attempt.
    let mut interval = tokio::time::interval(Duration::from_secs(reporter.interval_secs.max(1)));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            biased;
            deadline = stop.recv() => {
                if let Some(deadline) = deadline {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    let timeout = remaining.min(reporter.timeout);
                    reporter.deliver(true, timeout).await;
                }
                break;
            }
            _ = interval.tick() => {
                let timeout = reporter.timeout;
                reporter.deliver(false, timeout).await;
                interval = arm(reporter.interval_secs);
            }
            _ = notify.notified() => {
                if reporter.last_post.is_some_and(|at| at.elapsed() < Duration::from_secs(1)) {
                    continue;
                }
                let timeout = reporter.timeout;
                reporter.deliver(false, timeout).await;
                interval = arm(reporter.interval_secs);
            }
        }
    }
}

fn arm(secs: u64) -> tokio::time::Interval {
    let mut interval = tokio::time::interval(Duration::from_secs(secs.max(1)));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    interval.reset();
    interval
}

impl Reporter {
    fn try_new(parts: Parts) -> Option<Self> {
        if !activate(&parts.config) {
            return None;
        }
        let fingerprint = match parts.config.fingerprint() {
            Ok(fingerprint) => fingerprint,
            Err(err) => {
                tracing::warn!("{}", capped(&err.to_string()));
                return None;
            }
        };
        let client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Some(Self {
            secrets: known_secrets(&parts.config),
            interval_secs: parts.config.reporter.interval_secs.max(1),
            budget: parts.config.reporter.max_body_bytes,
            timeout: Duration::from_secs(parts.config.reporter.timeout_secs.max(1)),
            skew: Duration::from_secs(parts.config.reporter.token_skew_secs),
            mode: parts.mode,
            supervisor: parts.supervisor,
            registry: parts.registry,
            mcp: parts.mcp,
            booted: parts.booted,
            config: parts.config,
            instance_id: agentsmith_core::id::new_id(),
            fingerprint,
            client,
            sequence: 0,
            cached: None,
            disabled: false,
            last_post: None,
        })
    }

    #[cfg(test)]
    async fn cycle_once(&mut self) {
        let timeout = self.timeout;
        self.deliver(false, timeout).await;
    }

    async fn deliver(&mut self, stopping: bool, timeout: Duration) {
        if self.disabled || timeout.is_zero() {
            return;
        }
        let Some(mut token) = self.ensure_token(timeout).await else {
            return;
        };
        let mut draft = self.collect(stopping).await;
        let Some(mut prepared) = self.finish(&mut draft) else {
            return;
        };
        // One follow-up at most. A second failure waits for the next interval.
        let mut allow_retry = true;
        loop {
            self.last_post = Some(Instant::now());
            match self.post(&token, &prepared, timeout).await {
                Posted::Ok(body) => {
                    self.note_ack(&body, prepared.report_id);
                    return;
                }
                Posted::Unsupported => {
                    self.disable_protocol();
                    return;
                }
                Posted::Retry(delay) if allow_retry => {
                    tokio::time::sleep(delay).await;
                    allow_retry = false;
                    let Some(next) = self.ensure_token(timeout).await else {
                        return;
                    };
                    token = next;
                    draft = self.collect(stopping).await;
                    prepared = match self.finish(&mut draft) {
                        Some(prepared) => prepared,
                        None => return,
                    };
                }
                Posted::Auth if allow_retry => {
                    self.cached = None;
                    allow_retry = false;
                    let Some(next) = self.ensure_token(timeout).await else {
                        return;
                    };
                    token = next;
                    draft = self.collect(stopping).await;
                    prepared = match self.finish(&mut draft) {
                        Some(prepared) => prepared,
                        None => return,
                    };
                }
                Posted::TooLarge if allow_retry => {
                    self.send_smaller(&mut draft, stopping, prepared.stage, timeout, &token)
                        .await;
                    return;
                }
                Posted::Auth => {
                    self.cached = None;
                    return;
                }
                other => {
                    warn_posted(other);
                    return;
                }
            }
        }
    }

    async fn send_smaller(
        &mut self,
        draft: &mut Draft,
        stopping: bool,
        stage: u8,
        timeout: Duration,
        token: &str,
    ) {
        if stage >= SKELETON {
            return;
        }
        // Same runs. New id and sequence, then the next size step.
        let Some(prepared) = self.finish_at(draft, stopping, stage + 1) else {
            return;
        };
        self.last_post = Some(Instant::now());
        match self.post(token, &prepared, timeout).await {
            Posted::Ok(body) => self.note_ack(&body, prepared.report_id),
            Posted::Unsupported => self.disable_protocol(),
            Posted::Auth => self.cached = None,
            other => warn_posted(other),
        }
    }

    async fn ensure_token(&mut self, timeout: Duration) -> Option<String> {
        if let Some(cached) = &self.cached {
            if cached.expires_at > Instant::now() + self.skew {
                return Some(cached.access.clone());
            }
        }
        match self.fetch_token(timeout).await {
            Ok(cached) => {
                let access = cached.access.clone();
                self.cached = Some(cached);
                Some(access)
            }
            Err(err) => {
                tracing::warn!("AHSP token request failed: {err}");
                None
            }
        }
    }

    async fn fetch_token(&self, timeout: Duration) -> Result<CachedToken, String> {
        let form = form_body(&[
            ("grant_type", "client_credentials"),
            ("client_id", &self.config.reporter.client_id),
            ("client_secret", &self.config.reporter.client_secret),
        ]);
        let response = self
            .client
            .post(&self.config.reporter.token_url)
            .timeout(timeout)
            .header(
                reqwest::header::CONTENT_TYPE,
                "application/x-www-form-urlencoded",
            )
            .header(reqwest::header::USER_AGENT, USER_AGENT)
            .body(form)
            .send()
            .await
            .map_err(|err| capped(&err.to_string()))?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(format!("status {}", status.as_u16()));
        }
        let parsed: TokenJson =
            serde_json::from_str(&text).map_err(|_| "malformed token response".to_string())?;
        if parsed.access_token.is_empty() {
            return Err("malformed token response".to_string());
        }
        let Some(expires_in) = parsed.expires_in else {
            return Err("malformed token response".to_string());
        };
        Ok(CachedToken {
            access: parsed.access_token,
            expires_at: Instant::now() + Duration::from_secs(expires_in),
        })
    }

    async fn post(&self, token: &str, prepared: &Prepared, timeout: Duration) -> Posted {
        let response = self
            .client
            .post(&self.config.reporter.endpoint)
            .timeout(timeout)
            .header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"))
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header("Idempotency-Key", prepared.report_id.to_string())
            .header(reqwest::header::USER_AGENT, USER_AGENT)
            .body(prepared.body.clone())
            .send()
            .await;
        let response = match response {
            Ok(response) => response,
            Err(_) => return Posted::Retry(fallback_delay()),
        };
        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let text = response.text().await.unwrap_or_default();
        classify(status, retry_after.as_deref(), &text)
    }

    fn finish(&mut self, draft: &mut Draft) -> Option<Prepared> {
        let stopping = draft.report.harness.stopping;
        self.finish_at(draft, stopping, 0)
    }

    fn finish_at(&mut self, draft: &mut Draft, stopping: bool, stage: u8) -> Option<Prepared> {
        self.sequence += 1;
        stamp(&mut draft.report, self.sequence, stopping, self.booted);
        match render_from(draft, stage, self.budget, &self.secrets) {
            Some((body, stage)) => Some(Prepared {
                report_id: draft.report.report_id,
                body,
                stage,
            }),
            None => {
                self.sequence -= 1;
                None
            }
        }
    }

    fn note_ack(&mut self, body: &str, report_id: Uuid) {
        if let AckOutcome::Success {
            next_interval: Some(secs),
        } = parse_ack(body, report_id)
        {
            self.interval_secs = secs;
        }
    }

    fn disable_protocol(&mut self) {
        tracing::error!("AHSP server rejected protocol ahsp/1.0; reporter disabled");
        self.disabled = true;
    }

    async fn collect(&self, stopping: bool) -> Draft {
        let snaps = self.supervisor.snapshots();
        let (chosen, mut truncated) = select_runs(&snaps);
        let mut runs = Vec::with_capacity(chosen.len());
        let mut order = Vec::with_capacity(chosen.len());
        for item in &chosen {
            let (report, slot) = to_report(item, &mut truncated);
            runs.push(report);
            order.push(slot);
        }
        let (tools, tool_truncated) = tool_reports(&self.registry);
        let (mcp_servers, mcp_truncated) = mcp_reports(self.mcp.as_deref());
        let (host, host_truncated) = self.host().await;
        truncated |= tool_truncated || mcp_truncated || host_truncated;
        let report = ahsp::StateReport {
            protocol: ahsp::ProtocolId,
            report_id: agentsmith_core::id::new_id(),
            sequence: 1,
            sent_at: Utc::now(),
            harness: self.identity(stopping),
            host,
            runs,
            tools,
            mcp_servers,
            limits: limits_report(&self.config, self.supervisor.live_subagents()),
            truncated,
        };
        Draft { report, order }
    }

    fn identity(&self, stopping: bool) -> ahsp::HarnessIdentity {
        ahsp::HarnessIdentity {
            instance_id: self.instance_id,
            name: harness_name(&self.config),
            version: take_chars(env!("CARGO_PKG_VERSION"), 32).to_string(),
            pid: std::process::id(),
            process_mode: self.mode,
            uptime_secs: self.booted.elapsed().as_secs(),
            stopping,
            config_fingerprint: self.fingerprint.clone(),
        }
    }

    async fn host(&self) -> (ahsp::HostState, bool) {
        let (processes, truncated) = host_watch::sample(&self.config.host.watch).await;
        (
            ahsp::HostState {
                hostname: machine_hostname(),
                os: take_chars(std::env::consts::OS, 32).to_string(),
                arch: take_chars(std::env::consts::ARCH, 32).to_string(),
                processes,
            },
            truncated,
        )
    }
}

#[derive(serde::Deserialize)]
struct TokenJson {
    access_token: String,
    expires_in: Option<u64>,
}

fn classify(status: u16, retry_after: Option<&str>, body: &str) -> Posted {
    if (200..300).contains(&status) {
        return Posted::Ok(body.to_string());
    }
    if status == 401 || status == 403 {
        return Posted::Auth;
    }
    if status == 413 {
        return Posted::TooLarge;
    }
    if status == 422 && body.contains("unsupported_protocol") {
        return Posted::Unsupported;
    }
    if status == 429 || status == 503 {
        return Posted::Retry(retry_delay(retry_after));
    }
    Posted::GiveUp {
        status,
        body: body.to_string(),
    }
}

fn warn_posted(posted: Posted) {
    match posted {
        Posted::GiveUp { status, body } => {
            tracing::warn!("AHSP report failed with status {status}: {}", capped(&body));
        }
        Posted::TooLarge => tracing::warn!("AHSP report was rejected with status 413"),
        Posted::Retry(_) => tracing::warn!("AHSP report retry failed"),
        Posted::Auth => tracing::warn!("AHSP report was rejected after a token refresh"),
        Posted::Ok(_) | Posted::Unsupported => {}
    }
}

fn retry_delay(header: Option<&str>) -> Duration {
    let Some(header) = header.map(str::trim).filter(|text| !text.is_empty()) else {
        return fallback_delay();
    };
    if let Ok(secs) = header.parse::<u64>() {
        if secs <= 5 {
            return Duration::from_secs(secs);
        }
        return fallback_delay();
    }
    if let Ok(secs) = header.parse::<f64>() {
        if (0.0..=5.0).contains(&secs) {
            return Duration::from_secs_f64(secs);
        }
    }
    fallback_delay()
}

fn fallback_delay() -> Duration {
    Duration::from_secs(1) + jitter()
}

fn jitter() -> Duration {
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|elapsed| elapsed.subsec_nanos())
        .unwrap_or(0);
    Duration::from_millis(u64::from(nanos % 251))
}

fn form_body(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(key, value)| format!("{}={}", form_encode(key), form_encode(value)))
        .collect::<Vec<_>>()
        .join("&")
}

fn form_encode(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn known_secrets(config: &Config) -> Vec<String> {
    let mut secrets = Vec::new();
    push_secret(&mut secrets, &config.reporter.client_secret);
    push_secret(&mut secrets, &config.service.token);
    push_secret(&mut secrets, &config.llm.api_key);
    secrets
}

fn push_secret(out: &mut Vec<String>, value: &str) {
    if value.len() >= 8 && !out.iter().any(|have| have == value) {
        out.push(value.to_string());
    }
}

fn stamp(report: &mut ahsp::StateReport, sequence: u64, stopping: bool, booted: Instant) {
    report.report_id = agentsmith_core::id::new_id();
    report.sequence = sequence;
    report.sent_at = Utc::now();
    report.harness.uptime_secs = booted.elapsed().as_secs();
    report.harness.stopping = stopping;
    report.harness.pid = std::process::id();
}

fn render_from(draft: &Draft, stage: u8, budget: u64, secrets: &[String]) -> Option<(String, u8)> {
    let mut stage = stage;
    while stage <= SKELETON {
        let body = render(draft, stage, budget, secrets);
        if body.len() as u64 <= budget {
            return Some((body, stage));
        }
        stage += 1;
    }
    None
}

fn render(draft: &Draft, stage: u8, budget: u64, secrets: &[String]) -> String {
    let mut report = draft.report.clone();
    let mut order = draft.order.clone();
    if stage >= 1 {
        for process in &mut report.host.processes {
            process.running_instances.clear();
        }
        report.truncated = true;
    }
    if stage >= 2 {
        report.host.processes.clear();
        report.truncated = true;
    }
    if stage >= 3 {
        for run in &mut report.runs {
            run.children.clear();
        }
        report.truncated = true;
    }
    if stage >= 4 {
        for run in &mut report.runs {
            run.tool_use.clear();
        }
        report.truncated = true;
    }
    if stage >= 5 {
        for run in &mut report.runs {
            run.task_title = take_chars(&run.task_title, 32).to_string();
        }
        for server in &mut report.mcp_servers {
            if let Some(err) = server.last_error.clone() {
                server.last_error = Some(take_chars(&err, 32).to_string());
            }
        }
        report.truncated = true;
    }
    if stage == 6 {
        report.truncated = true;
        loop {
            let body = scrub_encode(&report, secrets);
            if body.len() as u64 <= budget {
                return body;
            }
            if !drop_one(&mut report, &mut order) {
                break;
            }
        }
        return scrub_encode(&report, secrets);
    }
    if stage >= SKELETON {
        report.runs.clear();
        report.tools.clear();
        report.mcp_servers.clear();
        report.host.processes.clear();
        report.truncated = true;
    }
    scrub_encode(&report, secrets)
}

fn drop_one(report: &mut ahsp::StateReport, order: &mut Vec<RunSlot>) -> bool {
    let Some(idx) = oldest_slot(order, true).or_else(|| oldest_slot(order, false)) else {
        return false;
    };
    let id = order[idx].id;
    order.remove(idx);
    report.runs.retain(|run| run.run_id != id);
    for run in &mut report.runs {
        run.children.retain(|child| child.run_id != id);
    }
    report.truncated = true;
    true
}

fn oldest_slot(order: &[RunSlot], descendant: bool) -> Option<usize> {
    order
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.terminal && if descendant { !slot.root } else { slot.root })
        .min_by(|(_, left), (_, right)| {
            finish_slot(left)
                .cmp(&finish_slot(right))
                .then(left.started.cmp(&right.started))
                .then(left.id.cmp(&right.id))
        })
        .map(|(idx, _)| idx)
}

fn finish_slot(slot: &RunSlot) -> SystemTime {
    slot.finished.unwrap_or(SystemTime::UNIX_EPOCH)
}

fn scrub_encode(report: &ahsp::StateReport, secrets: &[String]) -> String {
    let body = serde_json::to_string(report).unwrap_or_else(|_| "{}".to_string());
    scrub(&body, secrets)
}

fn scrub(body: &str, secrets: &[String]) -> String {
    let mut out = redact(body);
    for secret in secrets {
        if secret.len() >= 8 {
            out = out.replace(secret, "[redacted]");
        }
    }
    out
}

fn select_runs(snaps: &[RunSnapshot]) -> (Vec<Chosen>, bool) {
    let mut truncated = false;
    let mut live = Vec::new();
    let mut dead = Vec::new();
    for snap in snaps {
        if snap.parent_run_id.is_some() {
            continue;
        }
        if snap.status.is_terminal() {
            dead.push(snap);
        } else {
            live.push(snap);
        }
    }
    live.sort_by(|left, right| {
        right
            .started_at
            .cmp(&left.started_at)
            .then(right.run_id.cmp(&left.run_id))
    });
    dead.sort_by(|left, right| {
        finish_of(right)
            .cmp(&finish_of(left))
            .then(right.run_id.cmp(&left.run_id))
    });
    let mut roots = Vec::new();
    for snap in live.into_iter().chain(dead) {
        if roots.len() == MAX_ROOTS {
            truncated = true;
            break;
        }
        roots.push(snap.run_id);
    }
    let mut included: Vec<RunSnapshot> = snaps
        .iter()
        .filter(|snap| roots.contains(&snap.run_id) || roots.contains(&snap.root_run_id))
        .cloned()
        .collect();
    while included.len() > MAX_RUNS {
        let idx = oldest_terminal(&included, true).or_else(|| oldest_terminal(&included, false));
        let Some(idx) = idx else {
            break;
        };
        included.remove(idx);
        truncated = true;
    }
    included.sort_by(|left, right| {
        left.started_at
            .cmp(&right.started_at)
            .then(left.run_id.cmp(&right.run_id))
    });
    let mut chosen = Vec::with_capacity(included.len());
    for snap in &included {
        let mut kids: Vec<&RunSnapshot> = included
            .iter()
            .filter(|child| child.parent_run_id == Some(snap.run_id))
            .collect();
        kids.sort_by(|left, right| {
            right
                .started_at
                .cmp(&left.started_at)
                .then(right.run_id.cmp(&left.run_id))
        });
        if kids.len() > MAX_CHILDREN {
            truncated = true;
            kids.truncate(MAX_CHILDREN);
        }
        let children = kids
            .into_iter()
            .map(|kid| ahsp::ChildRef {
                run_id: kid.run_id,
                name: take_chars(&kid.name, 64).to_string(),
                status: map_status(kid.status),
                depth: kid.depth.min(4),
            })
            .collect();
        chosen.push(Chosen {
            snap: snap.clone(),
            children,
        });
    }
    (chosen, truncated)
}

fn oldest_terminal(runs: &[RunSnapshot], descendant: bool) -> Option<usize> {
    runs.iter()
        .enumerate()
        .filter(|(_, snap)| {
            let is_root = snap.parent_run_id.is_none();
            snap.status.is_terminal() && if descendant { !is_root } else { is_root }
        })
        .min_by(|(_, left), (_, right)| {
            finish_of(left)
                .cmp(&finish_of(right))
                .then(left.started_at.cmp(&right.started_at))
                .then(left.run_id.cmp(&right.run_id))
        })
        .map(|(idx, _)| idx)
}

fn finish_of(snap: &RunSnapshot) -> SystemTime {
    snap.finished_at.unwrap_or(SystemTime::UNIX_EPOCH)
}

fn to_report(chosen: &Chosen, truncated: &mut bool) -> (ahsp::RunReport, RunSlot) {
    let snap = &chosen.snap;
    let mut tool_use = snap.tool_use.clone();
    if tool_use.len() > MAX_TOOL_USE {
        *truncated = true;
        tool_use = tool_use.into_iter().take(MAX_TOOL_USE).collect();
    }
    let name = cut_field(&snap.name, 64, truncated);
    let depth = snap.depth.min(4);
    if depth != snap.depth {
        *truncated = true;
    }
    let spec_id = if snap.spec_id.is_empty() {
        None
    } else {
        Some(cut_field(&snap.spec_id, 64, truncated))
    };
    let error_kind = snap
        .error_kind
        .as_ref()
        .map(|kind| cut_field(kind, 32, truncated));
    let task_title = if snap.task_title.chars().count() > 120 {
        *truncated = true;
        agentsmith_core::redact::cut_display(&snap.task_title)
    } else {
        snap.task_title.clone()
    };
    let slot = RunSlot {
        id: snap.run_id,
        root: snap.parent_run_id.is_none(),
        terminal: snap.status.is_terminal(),
        started: snap.started_at,
        finished: snap.finished_at,
    };
    let report = ahsp::RunReport {
        run_id: snap.run_id,
        parent_run_id: snap.parent_run_id,
        root_run_id: snap.root_run_id,
        depth,
        name,
        mode: map_mode(snap.mode),
        status: map_status(snap.status),
        phase: map_phase(snap.phase),
        iteration: snap.iteration,
        max_iterations: snap.max_iterations.max(1),
        task_sha256_12: snap.task_sha256_12.clone(),
        task_title,
        spec_id,
        started_at: utc(snap.started_at),
        finished_at: snap.finished_at.map(utc),
        error_kind,
        children: chosen.children.clone(),
        tool_use,
    };
    (report, slot)
}

fn cut_field(text: &str, max_chars: usize, truncated: &mut bool) -> String {
    let cut = take_chars(text, max_chars);
    if cut.len() != text.len() {
        *truncated = true;
    }
    cut.to_string()
}

fn map_status(status: RunStatus) -> ahsp::RunStatus {
    match status {
        RunStatus::Running => ahsp::RunStatus::Running,
        RunStatus::Succeeded => ahsp::RunStatus::Succeeded,
        RunStatus::Failed => ahsp::RunStatus::Failed,
        RunStatus::Partial => ahsp::RunStatus::Partial,
        RunStatus::Cancelled => ahsp::RunStatus::Cancelled,
        RunStatus::MaxIterations => ahsp::RunStatus::MaxIterations,
        RunStatus::TimedOut => ahsp::RunStatus::TimedOut,
    }
}

fn map_phase(phase: Phase) -> ahsp::Phase {
    match phase {
        Phase::Plan => ahsp::Phase::Plan,
        Phase::React => ahsp::Phase::React,
        Phase::Join => ahsp::Phase::Join,
    }
}

fn map_mode(mode: LoopMode) -> ahsp::RunMode {
    match mode {
        LoopMode::React => ahsp::RunMode::React,
        LoopMode::Plan => ahsp::RunMode::Plan,
    }
}

fn utc(time: SystemTime) -> DateTime<Utc> {
    DateTime::<Utc>::from(time)
}

fn tool_reports(registry: &ToolRegistry) -> (Vec<ahsp::ToolReport>, bool) {
    let mut tools = Vec::new();
    let mut truncated = false;
    for name in registry.names() {
        if tools.len() == MAX_TOOLS {
            truncated = true;
            break;
        }
        let (source, server) = split_tool(&name);
        let report_name = cut_field(&name, 96, &mut truncated);
        let server = server.map(|server| cut_field(&server, 32, &mut truncated));
        tools.push(ahsp::ToolReport {
            name: report_name,
            source,
            server,
            enabled: true,
        });
    }
    (tools, truncated)
}

fn split_tool(name: &str) -> (ahsp::ToolSource, Option<String>) {
    match name.split_once("__") {
        Some((server, tool)) if !server.is_empty() && !tool.is_empty() => {
            (ahsp::ToolSource::Mcp, Some(server.to_string()))
        }
        _ => (ahsp::ToolSource::Builtin, None),
    }
}

fn mcp_reports(mcp: Option<&McpSupervisor>) -> (Vec<ahsp::McpServerReport>, bool) {
    let Some(mcp) = mcp else {
        return (Vec::new(), false);
    };
    let mut truncated = false;
    let mut servers = Vec::new();
    for snap in mcp.snapshot() {
        if servers.len() == MAX_SERVERS {
            truncated = true;
            break;
        }
        servers.push(mcp_report(snap, &mut truncated));
    }
    (servers, truncated)
}

fn mcp_report(snap: McpSnapshot, truncated: &mut bool) -> ahsp::McpServerReport {
    let status = match snap.status {
        McpSnapshotStatus::Up => ahsp::McpStatus::Up,
        McpSnapshotStatus::Down => ahsp::McpStatus::Down,
        McpSnapshotStatus::Connecting => ahsp::McpStatus::Connecting,
    };
    let last_error = snap.last_error.map(|err| cut_field(&err, 256, truncated));
    ahsp::McpServerReport {
        name: cut_field(&snap.name, 32, truncated),
        transport: ahsp::McpTransport::Stdio,
        command: cut_field(&snap.command, 64, truncated),
        status,
        required: snap.required,
        tool_count: snap.tool_count,
        tools_skipped: snap.tools_skipped,
        restart_count: snap.restart_count,
        connected_at: snap.connected_at.map(utc),
        last_error,
    }
}

fn limits_report(config: &Config, live: usize) -> ahsp::LimitsReport {
    ahsp::LimitsReport {
        max_subagent_depth: u32::try_from(config.loop_.max_subagent_depth).unwrap_or(u32::MAX),
        max_fanout: u32::try_from(config.loop_.max_fanout)
            .unwrap_or(u32::MAX)
            .max(1),
        live_subagents: u32::try_from(live).unwrap_or(u32::MAX),
    }
}

fn harness_name(config: &Config) -> String {
    let name = if config.reporter.harness_name.is_empty() {
        "agentsmith"
    } else {
        config.reporter.harness_name.as_str()
    };
    take_chars(name, 128).to_string()
}

fn machine_hostname() -> String {
    let raw = gethostname::gethostname();
    let name = raw.to_string_lossy();
    let name = name.trim();
    let name = if name.is_empty() { "unknown" } else { name };
    take_chars(name, 253).to_string()
}

fn parse_ack(body: &str, request_id: Uuid) -> AckOutcome {
    if body.trim().is_empty() {
        return AckOutcome::Success {
            next_interval: None,
        };
    }
    let mut value: Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(_) => {
            tracing::warn!("AHSP ack was not JSON: {}", capped(body));
            return AckOutcome::Success {
                next_interval: None,
            };
        }
    };
    if value.get("commands").is_some() {
        if let Some(object) = value.as_object_mut() {
            object.remove("commands");
        }
        warn_commands_once();
    }
    let ack: ahsp::Ack = match serde_json::from_value(value) {
        Ok(ack) => ack,
        Err(_) => {
            tracing::warn!("AHSP ack was not readable: {}", capped(body));
            return AckOutcome::Success {
                next_interval: None,
            };
        }
    };
    if let Some(protocol) = &ack.protocol {
        if !major_is_one(protocol) {
            tracing::error!("AHSP ack protocol is not major 1; continuing to send ahsp/1.0");
        }
    }
    if ack.report_id.is_some_and(|id| id != request_id) {
        tracing::warn!("AHSP ack report_id does not match the request");
    }
    if let Some(server_time) = ack.server_time {
        let delta = (Utc::now() - server_time).num_seconds().abs();
        if delta > CLOCK_SKEW_SECS && !CLOCK_WARNED.swap(true, Ordering::AcqRel) {
            tracing::warn!("AHSP server_time is more than 5 minutes from the local clock");
        }
    }
    if ack.accepted == Some(false) {
        tracing::warn!("AHSP report was not accepted: {}", capped(body));
        return AckOutcome::Rejected;
    }
    AckOutcome::Success {
        next_interval: ack.next_interval_secs.map(clamp_interval),
    }
}

fn warn_commands_once() {
    if !COMMANDS_WARNED.swap(true, Ordering::AcqRel) {
        COMMAND_WARNINGS.fetch_add(1, Ordering::Relaxed);
        tracing::warn!("AHSP 1.0 ignores control commands");
    }
}

fn major_is_one(protocol: &str) -> bool {
    let Some(rest) = protocol.strip_prefix("ahsp/") else {
        return false;
    };
    rest.split(['.', '/']).next() == Some("1")
}

pub(crate) fn clamp_interval(secs: u64) -> u64 {
    secs.clamp(10, 300)
}

fn capped(text: &str) -> String {
    take_chars(&redact(text), LOG_CHARS).to_string()
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::Mutex;
    use std::time::Duration;

    use agentsmith_core::redact::task_title;
    use agentsmith_protocol::{ProcessWatch, RunningInstance};
    use agentsmith_runtime::{LoopMode, Phase, RunStatus};
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use uuid::Uuid;

    use super::*;

    fn snap(
        n: u128,
        parent: Option<u128>,
        root: u128,
        status: RunStatus,
        started: u64,
        finished: Option<u64>,
    ) -> RunSnapshot {
        let at = |secs| SystemTime::UNIX_EPOCH + Duration::from_secs(secs);
        RunSnapshot {
            run_id: Uuid::from_u128(n),
            parent_run_id: parent.map(Uuid::from_u128),
            root_run_id: Uuid::from_u128(root),
            depth: u32::from(parent.is_some()),
            name: format!("run-{n}"),
            mode: LoopMode::React,
            status,
            phase: Phase::React,
            iteration: 0,
            max_iterations: 8,
            spec_id: "spec".into(),
            task_sha256_12: agentsmith_core::id::sha256_12(b"task"),
            task_title: "task".into(),
            started_at: at(started),
            finished_at: finished.map(at),
            error_kind: None,
            tool_use: BTreeMap::new(),
        }
    }

    #[test]
    fn more_than_64_runs_sets_truncated_and_keeps_a_running_run() {
        let mut snaps = vec![snap(1, None, 1, RunStatus::Running, 1, None)];
        for id in 2u128..=71 {
            snaps.push(snap(
                id,
                Some(1),
                1,
                RunStatus::Succeeded,
                id as u64,
                Some(id as u64),
            ));
        }
        snaps.push(snap(100, Some(1), 1, RunStatus::Running, 80, None));
        let (chosen, truncated) = select_runs(&snaps);
        assert!(truncated);
        assert_eq!(chosen.len(), 64);
        assert!(chosen
            .iter()
            .any(|run| run.snap.run_id == Uuid::from_u128(1)));
        assert!(chosen
            .iter()
            .any(|run| run.snap.run_id == Uuid::from_u128(100) && !run.snap.status.is_terminal()));
        assert!(chosen
            .iter()
            .all(|run| run.snap.run_id != Uuid::from_u128(2)));
        assert!(chosen
            .iter()
            .any(|run| run.snap.run_id == Uuid::from_u128(10)));
    }

    #[test]
    fn child_trim_keeps_the_newest_16() {
        let mut snaps = vec![snap(1, None, 1, RunStatus::Running, 1, None)];
        for id in 2u128..=21 {
            snaps.push(snap(
                id,
                Some(1),
                1,
                RunStatus::Succeeded,
                id as u64,
                Some(id as u64),
            ));
        }
        let (chosen, truncated) = select_runs(&snaps);
        assert!(truncated);
        assert_eq!(chosen.len(), 21);
        let root = chosen
            .iter()
            .find(|run| run.snap.run_id == Uuid::from_u128(1))
            .unwrap();
        assert_eq!(root.children.len(), 16);
        let ids: Vec<u128> = root
            .children
            .iter()
            .map(|child| child.run_id.as_u128())
            .collect();
        for id in 6..=21 {
            assert!(ids.contains(&id), "missing newest {id}");
        }
        for id in 2..6 {
            assert!(!ids.contains(&id), "kept oldest {id}");
        }
    }

    #[test]
    fn interval_hint_is_clamped() {
        assert_eq!(clamp_interval(1), 10);
        assert_eq!(clamp_interval(301), 300);
        assert_eq!(clamp_interval(30), 30);
    }

    #[test]
    fn planted_secret_is_absent_from_the_body() {
        let secret = "sk-plantedclientsecret";
        let mut config = Config::default();
        config.reporter.client_secret = secret.to_string();
        config.llm.api_key = secret.to_string();
        config.harness.workdir = std::path::PathBuf::from("/tmp/agentsmith-report");
        config.harness.data_dir = std::path::PathBuf::from("/tmp/agentsmith-report");
        let task = format!("{secret}\nfull task body {secret}");
        let mut run = snap(1, None, 1, RunStatus::Running, 1, None);
        run.task_title = task_title(&task);
        run.task_sha256_12 = agentsmith_core::id::sha256_12(task.as_bytes());
        let (chosen, _) = select_runs(&[run]);
        let (report_run, _) = to_report(&chosen[0], &mut false);
        let report = ahsp::StateReport {
            protocol: ahsp::ProtocolId,
            report_id: agentsmith_core::id::new_id(),
            sequence: 1,
            sent_at: Utc::now(),
            harness: ahsp::HarnessIdentity {
                instance_id: agentsmith_core::id::new_id(),
                name: "box".into(),
                version: "0.1.0".into(),
                pid: 1,
                process_mode: ahsp::ProcessMode::Foreground,
                uptime_secs: 1,
                stopping: false,
                config_fingerprint: "0123456789ab".into(),
            },
            host: ahsp::HostState {
                hostname: "box".into(),
                os: "linux".into(),
                arch: "x86_64".into(),
                processes: vec![ProcessWatch {
                    name: "tool".into(),
                    binary: "tool".into(),
                    installed: true,
                    version: None,
                    path: None,
                    running_instances: vec![RunningInstance {
                        pid: 9,
                        name: "tool".into(),
                        cmd_summary: host_watch::cmd_summary(&format!("tool --flag {secret}")),
                    }],
                }],
            },
            runs: vec![report_run],
            tools: Vec::new(),
            mcp_servers: Vec::new(),
            limits: ahsp::LimitsReport {
                max_subagent_depth: 1,
                max_fanout: 1,
                live_subagents: 0,
            },
            truncated: false,
        };
        let body = scrub_encode(&report, &known_secrets(&config));
        assert!(!body.contains(secret), "{body}");
        assert!(body.contains("[redacted]"), "{body}");
    }

    #[test]
    fn commands_are_ignored_once_and_ack_fields_are_read() {
        let id = agentsmith_core::id::new_id();
        let body = r#"{"accepted":true,"next_interval_secs":1,"commands":[{"op":"kill"}]}"#;
        let first = parse_ack(body, id);
        let second = parse_ack(body, id);
        assert!(matches!(
            first,
            AckOutcome::Success {
                next_interval: Some(10)
            }
        ));
        assert!(matches!(
            second,
            AckOutcome::Success {
                next_interval: Some(10)
            }
        ));
        assert_eq!(COMMAND_WARNINGS.load(Ordering::Relaxed), 1);
        let wide = r#"{"accepted":true,"next_interval_secs":301}"#;
        assert!(matches!(
            parse_ack(wide, id),
            AckOutcome::Success {
                next_interval: Some(300)
            }
        ));
    }

    #[test]
    fn enabled_with_an_empty_endpoint_does_not_start() {
        let mut config = Config::default();
        config.reporter.enabled = true;
        config.reporter.token_url = "http://127.0.0.1/token".into();
        config.reporter.client_id = "cid".into();
        config.reporter.client_secret = "secretsecret".into();
        config.reporter.harness_name = "box".into();
        let started = start(parts(config));
        assert!(started.is_none());
    }

    #[tokio::test]
    async fn token_failure_skips_the_report() {
        let mock = Mock::start().await;
        mock.enqueue(500, "{}", None);
        let mut reporter = reporter_for(&mock);
        reporter.cycle_once().await;
        let hits = mock.hits();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].path.ends_with("/token"));
    }

    #[tokio::test]
    async fn a_401_refreshes_once_and_does_not_loop() {
        let mock = Mock::start().await;
        mock.enqueue(200, &token("tok-1"), None);
        mock.enqueue(401, "{}", None);
        mock.enqueue(200, &token("tok-2"), None);
        mock.enqueue(401, "{}", None);
        let mut reporter = reporter_for(&mock);
        reporter.cycle_once().await;
        let hits = mock.hits();
        assert_eq!(hits.len(), 4, "{hits:?}");
        assert!(hits[0].path.ends_with("/token"));
        assert!(hits[1].path.ends_with("/state"));
        assert!(hits[2].path.ends_with("/token"));
        assert!(hits[3].path.ends_with("/state"));
        assert_eq!(hits[1].authorization, "Bearer tok-1");
        assert_eq!(hits[3].authorization, "Bearer tok-2");
        assert_eq!(hits[1].content_type, "application/json");
        assert_eq!(hits[1].user_agent, USER_AGENT);
        assert!(!hits[1].body.contains("sk-plantedclientsecret"));
        let first: Value = serde_json::from_str(&hits[1].body).unwrap();
        let second: Value = serde_json::from_str(&hits[3].body).unwrap();
        assert_eq!(first["sequence"], 1);
        assert_eq!(second["sequence"], 2);
        assert_ne!(first["report_id"], second["report_id"]);
    }

    #[tokio::test]
    async fn a_retry_uses_a_new_report_id_and_sequence() {
        let mock = Mock::start().await;
        mock.enqueue(200, &token("tok-1"), None);
        mock.enqueue(503, "{}", Some("0"));
        mock.enqueue(200, "{}", None);
        let mut reporter = reporter_for(&mock);
        reporter.cycle_once().await;
        let hits = mock.hits();
        let reports: Vec<_> = hits
            .iter()
            .filter(|hit| hit.path.ends_with("/state"))
            .collect();
        assert_eq!(reports.len(), 2, "{hits:?}");
        let first: Value = serde_json::from_str(&reports[0].body).unwrap();
        let second: Value = serde_json::from_str(&reports[1].body).unwrap();
        assert_eq!(first["sequence"], 1);
        assert_eq!(second["sequence"], 2);
        assert_ne!(first["report_id"], second["report_id"]);
        assert_eq!(reports[0].idempotency, first["report_id"].as_str().unwrap());
        assert_eq!(
            reports[1].idempotency,
            second["report_id"].as_str().unwrap()
        );
        assert_eq!(reports[0].content_type, "application/json");
        assert_eq!(reports[0].user_agent, USER_AGENT);
        assert!(!first["protocol"].is_null());
        assert!(first.get("commands").is_none());
    }

    #[test]
    fn retry_after_above_five_seconds_uses_the_fallback() {
        assert_eq!(retry_delay(Some("0")), Duration::ZERO);
        assert_eq!(retry_delay(Some("5")), Duration::from_secs(5));
        assert_eq!(retry_delay(Some("2.5")), Duration::from_secs_f64(2.5));
        let delay = retry_delay(Some("6"));
        assert!(delay >= Duration::from_secs(1));
        assert!(delay <= Duration::from_millis(1250));
        let http_date = retry_delay(Some("Wed, 21 Oct 2015 07:28:00 GMT"));
        assert!(http_date >= Duration::from_secs(1));
        assert!(http_date <= Duration::from_millis(1250));
    }

    fn token(access: &str) -> String {
        json!({ "access_token": access, "expires_in": 3600, "token_type": "bearer" }).to_string()
    }

    fn reporter_for(mock: &Mock) -> Reporter {
        let mut config = Config::default();
        config.reporter.enabled = true;
        config.reporter.endpoint = format!("http://{}/state", mock.addr);
        config.reporter.token_url = format!("http://{}/token", mock.addr);
        config.reporter.client_id = "cid".into();
        config.reporter.client_secret = "sk-plantedclientsecret".into();
        config.reporter.harness_name = "box".into();
        config.reporter.interval_secs = 30;
        config.reporter.timeout_secs = 2;
        config.reporter.token_skew_secs = 30;
        config.reporter.max_body_bytes = 262_144;
        config.harness.name = "box".into();
        config.harness.workdir = std::path::PathBuf::from("/tmp/agentsmith-report");
        config.harness.data_dir = std::path::PathBuf::from("/tmp/agentsmith-report");
        Reporter::try_new(parts(config)).expect("reporter")
    }

    fn parts(config: Config) -> Parts {
        Parts {
            config,
            mode: ahsp::ProcessMode::Service,
            supervisor: Arc::new(Supervisor::new()),
            registry: ToolRegistry::new(),
            mcp: None,
            booted: Instant::now(),
        }
    }

    #[derive(Clone, Debug)]
    struct Hit {
        path: String,
        authorization: String,
        content_type: String,
        user_agent: String,
        idempotency: String,
        body: String,
    }

    struct Scripted {
        status: u16,
        body: String,
        retry_after: Option<String>,
    }

    struct State {
        queue: Mutex<VecDeque<Scripted>>,
        hits: Mutex<Vec<Hit>>,
    }

    struct Mock {
        addr: std::net::SocketAddr,
        state: Arc<State>,
    }

    impl Mock {
        async fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
            let addr = listener.local_addr().expect("addr");
            let state = Arc::new(State {
                queue: Mutex::new(VecDeque::new()),
                hits: Mutex::new(Vec::new()),
            });
            let serve = Arc::clone(&state);
            tokio::spawn(async move {
                while let Ok((socket, _)) = listener.accept().await {
                    let state = Arc::clone(&serve);
                    tokio::spawn(async move {
                        let _ = handle_conn(socket, state).await;
                    });
                }
            });
            Self { addr, state }
        }

        fn enqueue(&self, status: u16, body: &str, retry_after: Option<&str>) {
            self.state.queue.lock().expect("queue").push_back(Scripted {
                status,
                body: body.to_string(),
                retry_after: retry_after.map(str::to_string),
            });
        }

        fn hits(&self) -> Vec<Hit> {
            self.state.hits.lock().expect("hits").clone()
        }
    }

    async fn handle_conn(mut socket: TcpStream, state: Arc<State>) -> std::io::Result<()> {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        let header_end = loop {
            let n = socket.read(&mut tmp).await?;
            if n == 0 {
                return Ok(());
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(pos) = buf.windows(4).position(|window| window == b"\r\n\r\n") {
                break pos;
            }
            if buf.len() > 1024 * 1024 {
                return Ok(());
            }
        };
        let header = String::from_utf8_lossy(&buf[..header_end]).into_owned();
        let mut lines = header.split("\r\n");
        let request = lines.next().unwrap_or("");
        let path = request.split_whitespace().nth(1).unwrap_or("").to_string();
        let mut content_length = 0usize;
        let mut authorization = String::new();
        let mut content_type = String::new();
        let mut user_agent = String::new();
        let mut idempotency = String::new();
        for line in lines {
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim();
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.parse().unwrap_or(0);
            } else if name.eq_ignore_ascii_case("authorization") {
                authorization = value.to_string();
            } else if name.eq_ignore_ascii_case("content-type") {
                content_type = value.to_string();
            } else if name.eq_ignore_ascii_case("user-agent") {
                user_agent = value.to_string();
            } else if name.eq_ignore_ascii_case("idempotency-key") {
                idempotency = value.to_string();
            }
        }
        let mut body = buf[header_end + 4..].to_vec();
        while body.len() < content_length {
            let n = socket.read(&mut tmp).await?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&tmp[..n]);
        }
        body.truncate(content_length);
        state.hits.lock().expect("hits").push(Hit {
            path,
            authorization,
            content_type,
            user_agent,
            idempotency,
            body: String::from_utf8_lossy(&body).into_owned(),
        });
        let spec = state.queue.lock().expect("queue").pop_front();
        let Some(spec) = spec else {
            return Ok(());
        };
        let mut head = format!(
            "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n",
            status = spec.status,
            len = spec.body.len(),
        );
        if let Some(retry) = &spec.retry_after {
            head.push_str(&format!("Retry-After: {retry}\r\n"));
        }
        head.push_str("\r\n");
        socket.write_all(head.as_bytes()).await?;
        socket.write_all(spec.body.as_bytes()).await?;
        let _ = socket.shutdown().await;
        Ok(())
    }
}
