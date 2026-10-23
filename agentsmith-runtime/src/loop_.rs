use std::path::PathBuf;
use std::sync::Arc;

use agentsmith_core::config::LoopConfig;
use agentsmith_core::Error;
use agentsmith_llm::{Llm, Message};

use crate::run_tools::RunTools;
use crate::spec::AgentSpec;
use crate::transcript::Transcript;
use crate::CancellationToken;

pub type RunId = uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    React,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Succeeded,
    Failed,
    Cancelled,
    MaxIterations,
}

pub struct RunOutcome {
    pub status: RunStatus,
    pub summary: String,
    /// Stop-table string. `None` when the run did not fail for one of those reasons.
    pub error_kind: Option<String>,
    /// Full conversation, including the system prompt and the task.
    pub transcript: Vec<Message>,
}

pub struct LoopLimits {
    pub react_max_iterations: u32,
    pub max_consecutive_tool_errors: u32,
}

impl LoopLimits {
    pub fn from_config(cfg: &LoopConfig) -> Result<Self, Error> {
        Ok(Self {
            react_max_iterations: u32_field("loop.react_max_iterations", cfg.react_max_iterations)?,
            max_consecutive_tool_errors: u32_field(
                "loop.max_consecutive_tool_errors",
                cfg.max_consecutive_tool_errors,
            )?,
        })
    }
}

fn u32_field(name: &str, value: u64) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::Config(format!("{name} does not fit in u32")))
}

/// Per-run inputs. The loop owns the transcript; the supervisor does not.
pub struct RunContext {
    pub run_id: RunId,
    pub spec: AgentSpec,
    pub task: String,
    pub extra_context: Option<String>,
    /// Resolved model. Empty spec model has already been replaced with `llm.model`.
    pub model: String,
    pub max_output_tokens: u32,
    pub temperature: Option<f32>,
    pub max_output_bytes: u64,
    pub workdir: PathBuf,
    pub llm: Arc<dyn Llm>,
    pub tools: RunTools,
    pub limits: LoopLimits,
    pub cancel: CancellationToken,
    pub events: Arc<dyn RunEvents>,
    pub transcript: Transcript,
}

#[async_trait::async_trait]
pub trait RunEvents: Send + Sync {
    async fn phase(&self, run: RunId, phase: Phase);
    async fn iteration(&self, run: RunId, n: u32);
    async fn finished(&self, run: RunId, status: RunStatus, error_kind: Option<String>);
}
