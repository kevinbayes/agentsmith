use std::collections::BTreeMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use agentsmith_core::{Error, ToolErrorKind};

use crate::tool::{tool_error, Tool, ToolDefinition};

/// Process-wide implementations. Not an allow-list.
///
/// The map is a `BTreeMap` so [`ToolRegistry::definitions`] is ordered by name.
/// `std::sync::RwLock` is intentional: the critical sections are synchronous.
/// `get` clones the `Arc` and drops the guard before the caller awaits
/// [`Tool::call`], so a tool can re-enter the registry without deadlocking.
#[derive(Clone)]
pub struct ToolRegistry {
    inner: Arc<RwLock<BTreeMap<String, Arc<dyn Tool>>>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(BTreeMap::new())),
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
        let mut map = self.write();
        if map.contains_key(&name) {
            return Err(tool_error(
                &name,
                ToolErrorKind::InvalidArgs,
                "duplicate tool name",
            ));
        }
        map.insert(name, tool);
        Ok(())
    }

    /// Drop one name. A re-list uses this so a foreign tool that shares the
    /// server prefix is left in place.
    pub fn unregister(&self, name: &str) -> bool {
        self.write().remove(name).is_some()
    }

    /// Drop every name that starts with `{prefix}__`. The prefix itself stays.
    pub fn unregister_prefix(&self, prefix: &str) {
        let marker = format!("{prefix}__");
        self.write().retain(|name, _| !name.starts_with(&marker));
    }

    /// Implementation lookup. The lock is not held after this returns.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.read().get(name).cloned()
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        // Collect under the lock, then ask each tool for its definition.
        // `definition` must not run while the guard is alive.
        let tools: Vec<Arc<dyn Tool>> = self.read().values().cloned().collect();
        tools.into_iter().map(|tool| tool.definition()).collect()
    }

    pub fn names(&self) -> Vec<String> {
        self.read().keys().cloned().collect()
    }

    fn read(&self) -> RwLockReadGuard<'_, BTreeMap<String, Arc<dyn Tool>>> {
        self.inner.read().unwrap_or_else(|err| err.into_inner())
    }

    fn write(&self) -> RwLockWriteGuard<'_, BTreeMap<String, Arc<dyn Tool>>> {
        self.inner.write().unwrap_or_else(|err| err.into_inner())
    }
}

/// One poll of external tool sessions. Both loops call this; reconnect stays
/// inside the implementer so the runtime does not own a transport.
#[async_trait::async_trait]
pub trait SessionPoll: Send + Sync {
    async fn poll(&self) -> Result<(), Error>;
}
