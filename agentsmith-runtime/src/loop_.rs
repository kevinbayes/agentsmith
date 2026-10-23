use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use agentsmith_core::config::{FailurePolicy, LoopConfig};
use agentsmith_core::redact::redact;
use agentsmith_core::Error;
use agentsmith_llm::{Llm, Message};
use agentsmith_tools::SessionPoll;

use crate::run_tools::RunTools;
use crate::spawn::SubagentSpawner;
use crate::spec::AgentSpec;
use crate::transcript::Transcript;
use crate::CancellationToken;

pub type RunId = uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Plan,
    React,
    Join,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Succeeded,
    Failed,
    Partial,
    Cancelled,
    MaxIterations,
    TimedOut,
}

impl RunStatus {
    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Running)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Partial => "partial",
            Self::Cancelled => "cancelled",
            Self::MaxIterations => "max_iterations",
            Self::TimedOut => "timed_out",
        }
    }
}

pub struct RunOutcome {
    pub status: RunStatus,
    pub summary: String,
    /// Stop-table string. `None` when the run did not fail for one of those reasons.
    pub error_kind: Option<String>,
    /// Full conversation, including the system prompt and the task.
    pub transcript: Vec<Message>,
}

#[derive(Clone)]
pub struct LoopLimits {
    pub react_max_iterations: u32,
    pub step_max_iterations: u32,
    pub planner_attempts: u8,
    pub max_consecutive_tool_errors: u32,
    pub max_subagent_depth: u32,
    pub max_fanout: usize,
    pub max_live_subagents: usize,
    pub spawn_timeout: Duration,
    pub failure_policy: FailurePolicy,
}

impl LoopLimits {
    pub fn from_config(cfg: &LoopConfig) -> Result<Self, Error> {
        Ok(Self {
            react_max_iterations: u32_field("loop.react_max_iterations", cfg.react_max_iterations)?,
            step_max_iterations: u32_field("loop.step_max_iterations", cfg.step_max_iterations)?,
            planner_attempts: u8_field("loop.planner_attempts", cfg.planner_attempts)?,
            max_consecutive_tool_errors: u32_field(
                "loop.max_consecutive_tool_errors",
                cfg.max_consecutive_tool_errors,
            )?,
            max_subagent_depth: u32_field("loop.max_subagent_depth", cfg.max_subagent_depth)?,
            max_fanout: usize_field("loop.max_fanout", cfg.max_fanout)?,
            max_live_subagents: usize_field("loop.max_live_subagents", cfg.max_live_subagents)?,
            spawn_timeout: Duration::from_secs(cfg.spawn_timeout_secs),
            failure_policy: cfg.failure_policy,
        })
    }
}

fn u32_field(name: &str, value: u64) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::Config(format!("{name} does not fit in u32")))
}

fn u8_field(name: &str, value: u64) -> Result<u8, Error> {
    u8::try_from(value).map_err(|_| Error::Config(format!("{name} does not fit in u8")))
}

fn usize_field(name: &str, value: u64) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| Error::Config(format!("{name} does not fit in usize")))
}

/// Identity recorded when a run is admitted. The supervisor copies it.
pub struct RunMeta {
    pub parent_run_id: Option<RunId>,
    pub depth: u32,
    pub name: String,
    pub max_iterations: u32,
    pub phase: Phase,
}

/// Per-run inputs. The loop owns the transcript; the supervisor does not.
pub struct RunContext {
    pub run_id: RunId,
    pub parent_run_id: Option<RunId>,
    /// Depth assigned to this run. Walker steps copy the walker. A model
    /// `spawn_subagents` call stores `caller.depth + 1`.
    pub depth: u32,
    pub name: String,
    pub spec: AgentSpec,
    pub task: String,
    pub extra_context: Option<String>,
    /// Resolved model. Empty spec model has already been replaced with `llm.model`.
    pub model: String,
    pub max_output_tokens: u32,
    pub temperature: Option<f32>,
    pub max_output_bytes: u64,
    /// Cap [`crate::ReactLoop`] reads. Root ReAct uses `react_max_iterations`.
    /// Plan steps and spawned children use `step_max_iterations`.
    pub max_iterations: u32,
    pub workdir: PathBuf,
    pub llm: Arc<dyn Llm>,
    pub tools: RunTools,
    pub limits: LoopLimits,
    pub cancel: CancellationToken,
    pub events: Arc<dyn RunEvents>,
    pub transcript: Transcript,
    /// `None` for a ReAct test that never spawns. Foreground runs always set it.
    pub spawner: Option<Arc<SubagentSpawner>>,
    /// MCP poller. `None` when no servers are configured.
    pub sessions: Option<Arc<dyn SessionPoll>>,
}

/// A down server is an observation on the next call, not a stop reason.
pub(crate) async fn poll_sessions(sessions: &Option<Arc<dyn SessionPoll>>) {
    let Some(sessions) = sessions else {
        return;
    };
    if let Err(err) = sessions.poll().await {
        eprintln!("agentsmith: mcp poll: {}", redact(&err.to_string()));
    }
}

#[async_trait::async_trait]
pub trait RunEvents: Send + Sync {
    async fn started(&self, run: RunId, meta: RunMeta);
    async fn phase(&self, run: RunId, phase: Phase);
    async fn iteration(&self, run: RunId, n: u32);
    async fn tool_used(&self, run: RunId, name: &str);
    /// First terminal status wins. The return value is the status that stuck,
    /// so a late caller can write that status instead of a second one.
    async fn finished(
        &self,
        run: RunId,
        status: RunStatus,
        error_kind: Option<String>,
    ) -> (RunStatus, Option<String>);
    /// Same rule as [`Self::finished`], for a `Drop` during unwind that cannot await.
    fn record_terminal(
        &self,
        run: RunId,
        status: RunStatus,
        error_kind: Option<String>,
    ) -> (RunStatus, Option<String>);
}
