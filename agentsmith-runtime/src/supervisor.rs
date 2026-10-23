use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use crate::loop_::{Phase, RunEvents, RunId, RunMeta, RunStatus};
use crate::snapshot::RunSnapshot;

#[derive(Debug)]
struct RunState {
    parent_run_id: Option<RunId>,
    depth: u32,
    name: String,
    status: RunStatus,
    phase: Phase,
    iteration: u32,
    max_iterations: u32,
    error_kind: Option<String>,
    tool_use: BTreeMap<String, u64>,
}

impl RunState {
    fn running() -> Self {
        Self {
            parent_run_id: None,
            depth: 0,
            name: String::new(),
            status: RunStatus::Running,
            phase: Phase::React,
            iteration: 0,
            max_iterations: 0,
            error_kind: None,
            tool_use: BTreeMap::new(),
        }
    }

    fn apply_meta(&mut self, meta: RunMeta) {
        self.parent_run_id = meta.parent_run_id;
        self.depth = meta.depth;
        self.name = meta.name;
        self.max_iterations = meta.max_iterations;
        self.phase = meta.phase;
    }

    fn snapshot(&self, run_id: RunId) -> RunSnapshot {
        RunSnapshot {
            run_id,
            parent_run_id: self.parent_run_id,
            depth: self.depth,
            name: self.name.clone(),
            status: self.status,
            phase: self.phase,
            iteration: self.iteration,
            max_iterations: self.max_iterations,
            error_kind: self.error_kind.clone(),
            tool_use: self.tool_use.clone(),
        }
    }
}

/// In-memory runs. The transcript stays on the loop, not here.
///
/// The first terminal status sticks. A detached child can still finish after
/// its join has already recorded `timed_out`, and that late exit must not
/// overwrite it.
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
        Some(state.snapshot(run_id))
    }

    pub fn snapshots(&self) -> Vec<RunSnapshot> {
        let runs = lock(&self.runs);
        runs.iter()
            .map(|(run_id, state)| state.snapshot(*run_id))
            .collect()
    }

    /// Drop a terminal `root` and its terminal descendants.
    ///
    /// Non-terminal descendants stay so an in-flight child is not dropped with its root.
    pub fn remove_terminal_tree(&self, root: RunId) -> Vec<RunId> {
        let mut runs = lock(&self.runs);
        let mut tree = vec![root];
        let mut index = 0;
        while index < tree.len() {
            let id = tree[index];
            for (child, state) in runs.iter() {
                if state.parent_run_id == Some(id) && !tree.contains(child) {
                    tree.push(*child);
                }
            }
            index += 1;
        }
        let mut removed = Vec::new();
        for id in tree {
            let terminal = runs
                .get(&id)
                .is_some_and(|state| state.status.is_terminal());
            if terminal {
                runs.remove(&id);
                removed.push(id);
            }
        }
        removed
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}

#[async_trait::async_trait]
impl RunEvents for Supervisor {
    async fn started(&self, run: RunId, meta: RunMeta) {
        let mut runs = lock(&self.runs);
        let state = runs.entry(run).or_insert_with(RunState::running);
        if state.status.is_terminal() {
            return;
        }
        state.apply_meta(meta);
    }

    async fn phase(&self, run: RunId, phase: Phase) {
        let mut runs = lock(&self.runs);
        let state = runs.entry(run).or_insert_with(RunState::running);
        if state.status.is_terminal() {
            return;
        }
        state.phase = phase;
    }

    async fn iteration(&self, run: RunId, n: u32) {
        let mut runs = lock(&self.runs);
        let state = runs.entry(run).or_insert_with(RunState::running);
        if state.status.is_terminal() {
            return;
        }
        state.iteration = n;
    }

    async fn tool_used(&self, run: RunId, name: &str) {
        let mut runs = lock(&self.runs);
        let state = runs.entry(run).or_insert_with(RunState::running);
        if state.status.is_terminal() {
            return;
        }
        *state.tool_use.entry(name.to_string()).or_insert(0) += 1;
    }

    async fn finished(
        &self,
        run: RunId,
        status: RunStatus,
        error_kind: Option<String>,
    ) -> (RunStatus, Option<String>) {
        self.record_terminal(run, status, error_kind)
    }

    fn record_terminal(
        &self,
        run: RunId,
        status: RunStatus,
        error_kind: Option<String>,
    ) -> (RunStatus, Option<String>) {
        let mut runs = lock(&self.runs);
        let state = runs.entry(run).or_insert_with(RunState::running);
        if !state.status.is_terminal() {
            state.status = status;
            state.error_kind = error_kind;
        }
        (state.status, state.error_kind.clone())
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

    #[tokio::test]
    async fn a_late_finish_does_not_overwrite_timed_out() {
        let supervisor = Supervisor::new();
        let id = agentsmith_core::id::new_id();
        supervisor
            .started(
                id,
                RunMeta {
                    parent_run_id: None,
                    depth: 1,
                    name: "reader".into(),
                    max_iterations: 8,
                    phase: Phase::React,
                },
            )
            .await;
        supervisor
            .finished(id, RunStatus::TimedOut, Some("timeout".into()))
            .await;
        let stuck = supervisor.finished(id, RunStatus::Cancelled, None).await;
        assert_eq!(stuck, (RunStatus::TimedOut, Some("timeout".into())));
        supervisor.iteration(id, 4).await;
        let snap = supervisor.snapshot(id).unwrap();
        assert_eq!(snap.status, RunStatus::TimedOut);
        assert_eq!(snap.error_kind.as_deref(), Some("timeout"));
        assert_eq!(snap.depth, 1);
        assert_eq!(snap.iteration, 0);
        assert_eq!(snap.max_iterations, 8);
    }

    #[test]
    fn remove_terminal_tree_drops_terminal_descendants_and_keeps_live_ones() {
        let supervisor = Supervisor::new();
        let root = agentsmith_core::id::new_id();
        let child = agentsmith_core::id::new_id();
        let live = agentsmith_core::id::new_id();
        supervisor.record_terminal(root, RunStatus::Failed, Some("mcp_down".into()));
        {
            let mut runs = lock(&supervisor.runs);
            runs.insert(
                child,
                RunState {
                    parent_run_id: Some(root),
                    status: RunStatus::Succeeded,
                    ..RunState::running()
                },
            );
            runs.insert(
                live,
                RunState {
                    parent_run_id: Some(root),
                    status: RunStatus::Running,
                    ..RunState::running()
                },
            );
        }
        let removed = supervisor.remove_terminal_tree(root);
        assert!(removed.contains(&root));
        assert!(removed.contains(&child));
        assert!(!removed.contains(&live));
        assert!(supervisor.snapshot(root).is_none());
        assert!(supervisor.snapshot(child).is_none());
        assert_eq!(
            supervisor.snapshot(live).unwrap().status,
            RunStatus::Running
        );
        let kept = agentsmith_core::id::new_id();
        supervisor.record_terminal(kept, RunStatus::TimedOut, Some("timeout".into()));
        let stuck = supervisor.record_terminal(kept, RunStatus::Cancelled, None);
        assert_eq!(stuck, (RunStatus::TimedOut, Some("timeout".into())));
    }
}
