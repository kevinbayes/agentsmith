//! ReAct loop, per-run tool view, and local JSONL transcripts.

mod foreground;
mod jsonl;
mod loop_;
mod react;
mod run_tools;
mod snapshot;
mod spec;
mod supervisor;
mod transcript;

pub use foreground::{run_foreground, ForegroundParams};
pub use loop_::{
    AgentLoop, LoopLimits, Phase, RunContext, RunEvents, RunId, RunOutcome, RunStatus,
};
pub use react::ReactLoop;
pub use run_tools::{RunKind, RunTools};
pub use snapshot::RunSnapshot;
pub use spec::{load_spec, resolve_mode, AgentSpec, LoopMode};
pub use supervisor::Supervisor;
pub use tokio_util::sync::CancellationToken;
pub use transcript::Transcript;
