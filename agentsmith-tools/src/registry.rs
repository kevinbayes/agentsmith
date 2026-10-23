use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use agentsmith_core::{Error, ToolErrorKind};

use crate::tool::{tool_error, Tool, ToolDefinition};

struct State {
    tools: BTreeMap<String, Arc<dyn Tool>>,
    /// Names the supervisor dropped because it marked that server down.
    /// A re-list removal does not land here.
    down: BTreeSet<String>,
}

/// Process-wide implementations. Not an allow-list.
///
/// The map is a `BTreeMap` so [`ToolRegistry::definitions`] is ordered by name.
/// `std::sync::RwLock` is intentional: the critical sections are synchronous.
/// `get` clones the `Arc` and drops the guard before the caller awaits
/// [`Tool::call`], so a tool can re-enter the registry without deadlocking.
#[derive(Clone)]
pub struct ToolRegistry {
    inner: Arc<RwLock<State>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(State {
                tools: BTreeMap::new(),
                down: BTreeSet::new(),
            })),
        }
    }

    /// Insert `tool` under `definition().name`. A duplicate name is rejected.
    pub fn register(&self, tool: Arc<dyn Tool>) -> Result<(), Error> {
        // Resolve the name before taking the write lock so `definition` can
        // re-enter the registry.
        let name = tool.definition().name;
        if name.is_empty() {
            return Err(tool_error(
                "tool",
                ToolErrorKind::InvalidArgs,
                "tool name is empty",
            ));
        }
        let mut state = self.write();
        if state.tools.contains_key(&name) {
            return Err(tool_error(
                &name,
                ToolErrorKind::InvalidArgs,
                "duplicate tool name",
            ));
        }
        state.tools.insert(name, tool);
        Ok(())
    }

    /// Drop one name. A re-list uses this so a foreign tool that shares the
    /// server prefix is left in place. This does not mark the name down.
    pub fn unregister(&self, name: &str) -> bool {
        self.write().tools.remove(name).is_some()
    }

    /// Record `name` as a server the supervisor has marked down.
    pub fn mark_mcp_down(&self, name: &str) {
        self.write().down.insert(name.to_string());
    }

    /// The server is up again, so those names are no longer `mcp_down`.
    pub fn clear_mcp_down<'a>(&self, names: impl IntoIterator<Item = &'a str>) {
        let mut state = self.write();
        for name in names {
            state.down.remove(name);
        }
    }

    pub fn is_mcp_down(&self, name: &str) -> bool {
        self.read().down.contains(name)
    }

    /// Drop every name that starts with `{prefix}__`. The prefix itself stays.
    pub fn unregister_prefix(&self, prefix: &str) {
        let marker = format!("{prefix}__");
        self.write()
            .tools
            .retain(|name, _| !name.starts_with(&marker));
    }

    /// Implementation lookup. The lock is not held after this returns.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.read().tools.get(name).cloned()
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        // Collect under the lock, then ask each tool for its definition.
        // `definition` must not run while the guard is alive.
        let tools: Vec<Arc<dyn Tool>> = self.read().tools.values().cloned().collect();
        tools.into_iter().map(|tool| tool.definition()).collect()
    }

    pub fn names(&self) -> Vec<String> {
        self.read().tools.keys().cloned().collect()
    }

    fn read(&self) -> RwLockReadGuard<'_, State> {
        self.inner.read().unwrap_or_else(|err| err.into_inner())
    }

    fn write(&self) -> RwLockWriteGuard<'_, State> {
        self.inner.write().unwrap_or_else(|err| err.into_inner())
    }
}

/// One poll of external tool sessions. Both loops call this; reconnect stays
/// inside the implementer so the runtime does not own a transport.
#[async_trait::async_trait]
pub trait SessionPoll: Send + Sync {
    async fn poll(&self) -> Result<(), Error>;
}
