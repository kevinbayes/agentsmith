use std::collections::BTreeMap;
use std::time::SystemTime;

use crate::loop_::{Phase, RunId, RunStatus};
use crate::spec::LoopMode;

/// Fields the supervisor can hand out without the transcript.
///
/// The task body is not here. `task_sha256_12` and `task_title` are computed
/// when the run is recorded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunSnapshot {
    pub run_id: RunId,
    pub parent_run_id: Option<RunId>,
    pub root_run_id: RunId,
    pub depth: u32,
    pub name: String,
    pub mode: LoopMode,
    pub status: RunStatus,
    pub phase: Phase,
    pub iteration: u32,
    pub max_iterations: u32,
    pub spec_id: String,
    pub task_sha256_12: String,
    pub task_title: String,
    pub started_at: SystemTime,
    pub finished_at: Option<SystemTime>,
    pub error_kind: Option<String>,
    pub tool_use: BTreeMap<String, u64>,
}
