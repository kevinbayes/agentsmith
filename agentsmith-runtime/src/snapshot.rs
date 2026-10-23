use std::collections::BTreeMap;

use crate::loop_::{Phase, RunId, RunStatus};

/// Fields the supervisor can hand out without the transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunSnapshot {
    pub run_id: RunId,
    pub parent_run_id: Option<RunId>,
    pub depth: u32,
    pub name: String,
    pub status: RunStatus,
    pub phase: Phase,
    pub iteration: u32,
    pub max_iterations: u32,
    pub error_kind: Option<String>,
    pub tool_use: BTreeMap<String, u64>,
}
