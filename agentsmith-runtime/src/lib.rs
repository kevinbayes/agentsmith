//! ReAct loop, plan walker, per-run tool view, and local JSONL transcripts.

mod foreground;
mod jsonl;
mod loop_;
mod plan;
mod react;
mod run_tools;
mod snapshot;
mod spawn;
mod spec;
mod supervisor;
mod transcript;

pub use foreground::{run_admitted, run_foreground, AdmittedRun, ForegroundParams};
pub use loop_::{LoopLimits, Phase, RunContext, RunEvents, RunId, RunMeta, RunOutcome, RunStatus};
pub use plan::PlanExecuteLoop;
pub use react::ReactLoop;
pub use run_tools::{RunKind, RunTools, ToolFilter};
pub use snapshot::RunSnapshot;
pub use spawn::{
    ChildOrigin, ChildOutcome, ChildSpec, JoinOutcome, SpawnHub, SpawnRequest, SpawnerConfig,
    SubagentSpawner,
};
pub use spec::{load_spec, resolve_mode, AgentSpec, LoopMode};
pub use supervisor::Supervisor;
pub use tokio_util::sync::CancellationToken;
pub use transcript::Transcript;
