use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use crate::loop_::{Phase, RunEvents, RunId, RunStatus};
use crate::snapshot::RunSnapshot;

#[derive(Debug)]
struct RunState {
    status: RunStatus,
    phase: Phase,
    iteration: u32,
    error_kind: Option<String>,
}

impl RunState {
    fn running() -> Self {
        Self {
            status: RunStatus::Running,
            phase: Phase::React,
            iteration: 0,
            error_kind: None,
        }
    }
}

/// In-memory runs. The transcript stays on the loop, not here.
#[derive(Debug, Default)]
pub struct Supervisor {
    runs: Mutex<BTreeMap<RunId, RunState>>,
}

impl Supervisor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self, run_id: RunId) -> Option<RunSnapshot> {
        let runs = lock(&self.runs);
        let state = runs.get(&run_id)?;
        Some(RunSnapshot {
            run_id,
            status: state.status,
            phase: state.phase,
            iteration: state.iteration,
            error_kind: state.error_kind.clone(),
        })
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}

#[async_trait::async_trait]
impl RunEvents for Supervisor {
    async fn phase(&self, run: RunId, phase: Phase) {
        lock(&self.runs)
            .entry(run)
            .or_insert_with(RunState::running)
            .phase = phase;
    }

    async fn iteration(&self, run: RunId, n: u32) {
        lock(&self.runs)
            .entry(run)
            .or_insert_with(RunState::running)
            .iteration = n;
    }

    async fn finished(&self, run: RunId, status: RunStatus, error_kind: Option<String>) {
        let mut runs = lock(&self.runs);
        let state = runs.entry(run).or_insert_with(RunState::running);
        state.status = status;
        state.error_kind = error_kind;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn snapshot_is_a_copy_of_the_status_fields() {
        let supervisor = Supervisor::new();
        let id = agentsmith_core::id::new_id();
        supervisor.phase(id, Phase::React).await;
        supervisor.iteration(id, 3).await;
        let snap = supervisor.snapshot(id).unwrap();
        supervisor
            .finished(id, RunStatus::Failed, Some("length".into()))
            .await;
        assert_eq!(snap.status, RunStatus::Running);
        assert_eq!(snap.phase, Phase::React);
        assert_eq!(snap.iteration, 3);
        assert!(snap.error_kind.is_none());
        let later = supervisor.snapshot(id).unwrap();
        assert_eq!(later.status, RunStatus::Failed);
        assert_eq!(later.error_kind.as_deref(), Some("length"));
        assert_eq!(later.iteration, 3);
        assert!(supervisor.snapshot(agentsmith_core::id::new_id()).is_none());
    }
}
