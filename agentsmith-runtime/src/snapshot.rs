use crate::loop_::{Phase, RunId, RunStatus};

/// Fields the supervisor can hand out without the transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunSnapshot {
    pub run_id: RunId,
    pub status: RunStatus,
    pub phase: Phase,
    pub iteration: u32,
    pub error_kind: Option<String>,
}
