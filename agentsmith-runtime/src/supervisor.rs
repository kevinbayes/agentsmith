use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::SystemTime;

use tokio::sync::Notify;

use crate::loop_::{Phase, RunEvents, RunId, RunMeta, RunStatus};
use crate::snapshot::RunSnapshot;
use crate::spec::LoopMode;

#[derive(Debug)]
struct RunState {
    parent_run_id: Option<RunId>,
    root_run_id: RunId,
    depth: u32,
    name: String,
    mode: LoopMode,
    status: RunStatus,
    phase: Phase,
    iteration: u32,
    max_iterations: u32,
    spec_id: String,
    task_sha256_12: String,
    task_title: String,
    started_at: SystemTime,
    finished_at: Option<SystemTime>,
    error_kind: Option<String>,
    tool_use: BTreeMap<String, u64>,
}

impl RunState {
    fn running() -> Self {
        Self {
            parent_run_id: None,
            root_run_id: RunId::nil(),
            depth: 0,
            name: String::new(),
            mode: LoopMode::React,
            status: RunStatus::Running,
            phase: Phase::React,
            iteration: 0,
            max_iterations: 0,
            spec_id: String::new(),
            task_sha256_12: agentsmith_core::id::sha256_12(b""),
            task_title: String::new(),
            started_at: SystemTime::now(),
            finished_at: None,
            error_kind: None,
            tool_use: BTreeMap::new(),
        }
    }

    fn apply_meta(&mut self, meta: RunMeta, root: RunId) {
        self.parent_run_id = meta.parent_run_id;
        self.depth = meta.depth;
        self.name = meta.name;
        self.max_iterations = meta.max_iterations;
        self.phase = meta.phase;
        self.mode = meta.mode;
        self.spec_id = meta.spec_id;
        // A second `started` (service admit, then the loop) must keep the first clock.
        if self.root_run_id.is_nil() {
            self.root_run_id = root;
        }
        if !meta.task.is_empty() || self.task_title.is_empty() {
            self.task_sha256_12 = agentsmith_core::id::sha256_12(meta.task.as_bytes());
            self.task_title = agentsmith_core::redact::task_title(&meta.task);
        }
    }

    fn snapshot(&self, run_id: RunId) -> RunSnapshot {
        let root_run_id = if self.root_run_id.is_nil() {
            run_id
        } else {
            self.root_run_id
        };
        RunSnapshot {
            run_id,
            parent_run_id: self.parent_run_id,
            root_run_id,
            depth: self.depth,
            name: self.name.clone(),
            mode: self.mode,
            status: self.status,
            phase: self.phase,
            iteration: self.iteration,
            max_iterations: self.max_iterations,
            spec_id: self.spec_id.clone(),
            task_sha256_12: self.task_sha256_12.clone(),
            task_title: self.task_title.clone(),
            started_at: self.started_at,
            finished_at: self.finished_at,
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
#[derive(Debug)]
pub struct Supervisor {
    runs: Mutex<BTreeMap<RunId, RunState>>,
    live: AtomicUsize,
    root_terminal: Arc<Notify>,
}

impl Supervisor {
    pub fn new() -> Self {
        Self {
            runs: Mutex::new(BTreeMap::new()),
            live: AtomicUsize::new(0),
            root_terminal: Arc::new(Notify::new()),
        }
    }
}

impl Default for Supervisor {
    fn default() -> Self {
        Self::new()
    }
}

impl Supervisor {
    pub fn live_subagents(&self) -> usize {
        self.live.load(Ordering::Acquire)
    }

    pub fn add_live(&self, delta: isize) {
        if delta > 0 {
            self.live.fetch_add(delta as usize, Ordering::AcqRel);
            return;
        }
        if delta == 0 {
            return;
        }
        let n = delta.unsigned_abs();
        let mut current = self.live.load(Ordering::Acquire);
        loop {
            let next = current.saturating_sub(n);
            match self.live.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    /// Wakes a reporter when a root run first reaches a terminal status.
    pub fn root_terminal(&self) -> Arc<Notify> {
        Arc::clone(&self.root_terminal)
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
        let root = match meta.parent_run_id {
            None => run,
            Some(parent) => match runs.get(&parent).map(|state| state.root_run_id) {
                Some(root) if !root.is_nil() => root,
                _ => parent,
            },
        };
        let state = runs.entry(run).or_insert_with(RunState::running);
        if state.status.is_terminal() {
            return;
        }
        // `or_insert` already stamped `started_at`. Do not move it.
        state.apply_meta(meta, root);
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
        let (result, notify) = {
            let mut runs = lock(&self.runs);
            let state = runs.entry(run).or_insert_with(RunState::running);
            let mut notify = false;
            if !state.status.is_terminal() {
                state.status = status;
                state.error_kind = error_kind;
                state.finished_at = Some(SystemTime::now());
                notify = state.parent_run_id.is_none();
            }
            ((state.status, state.error_kind.clone()), notify)
        };
        if notify {
            self.root_terminal.notify_one();
        }
        result
    }

    fn adjust_live(&self, delta: isize) {
        self.add_live(delta);
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
                    mode: LoopMode::React,
                    spec_id: "reader".into(),
                    task: "sk-testsecret".into(),
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
        assert_eq!(snap.task_title, "[redacted]");
        assert!(!snap.task_title.contains("testsecret"));
        assert_eq!(snap.task_sha256_12.len(), 12);
        assert!(snap.finished_at.is_some());
        let started_at = snap.started_at;
        supervisor
            .started(
                id,
                RunMeta {
                    parent_run_id: None,
                    depth: 1,
                    name: "reader".into(),
                    max_iterations: 8,
                    phase: Phase::React,
                    mode: LoopMode::React,
                    spec_id: "reader".into(),
                    task: "a later task".into(),
                },
            )
            .await;
        let again = supervisor.snapshot(id).unwrap();
        assert_eq!(again.started_at, started_at);
        assert_eq!(again.task_title, "[redacted]");
        assert_eq!(again.status, RunStatus::TimedOut);
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
