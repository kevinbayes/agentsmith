use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use agentsmith_core::config::McpServerConfig;
use agentsmith_core::redact::redact;
use agentsmith_core::{Error, ToolErrorKind};
use agentsmith_tools::{SessionPoll, Tool, ToolCtx, ToolDefinition, ToolOutput, ToolRegistry};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::install_tools::{install_tools, PreparedTool};
use crate::stdio::StdioMcpClient;

const BACKOFF_START: Duration = Duration::from_secs(1);
const BACKOFF_CAP: Duration = Duration::from_secs(300);

pub struct McpSupervisorConfig {
    pub registry: ToolRegistry,
    pub servers: BTreeMap<String, McpServerConfig>,
    pub max_output_bytes: u64,
    pub shutdown_grace: Duration,
    pub cancel: CancellationToken,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Up,
    Down,
}

struct ServerSlot {
    required: bool,
    client: StdioMcpClient,
    status: Status,
    /// Next wait after a failed connect. Starts at 1s and doubles, capped at 300s.
    backoff: Duration,
    next_attempt: Option<Instant>,
    registered: BTreeSet<String>,
    /// Names this server marked down. Cleared once the server is up again.
    down_names: BTreeSet<String>,
    tools_skipped: u32,
}

/// Owns every stdio session. `poll` is the only connect, re-list, and backoff site.
pub struct McpSupervisor {
    registry: ToolRegistry,
    servers: Mutex<BTreeMap<String, ServerSlot>>,
    grace: Duration,
    weak: Weak<Self>,
    /// Serializes `poll` so two loops cannot spawn the same server twice.
    poll_lock: tokio::sync::Mutex<()>,
    first_poll: AtomicBool,
}

impl McpSupervisor {
    pub fn new(config: McpSupervisorConfig) -> Arc<Self> {
        Arc::new_cyclic(|weak| {
            let mut servers = BTreeMap::new();
            for (name, server) in config.servers {
                let client = StdioMcpClient::new(
                    &name,
                    &server,
                    config.max_output_bytes,
                    config.shutdown_grace,
                    config.cancel.clone(),
                );
                servers.insert(
                    name,
                    ServerSlot {
                        required: server.required,
                        client,
                        status: Status::Down,
                        backoff: BACKOFF_START,
                        next_attempt: None,
                        registered: BTreeSet::new(),
                        down_names: BTreeSet::new(),
                        tools_skipped: 0,
                    },
                );
            }
            Self {
                registry: config.registry,
                servers: Mutex::new(servers),
                grace: config.shutdown_grace,
                weak: weak.clone(),
                poll_lock: tokio::sync::Mutex::new(()),
                first_poll: AtomicBool::new(true),
            }
        })
    }

    /// Close every session, sharing one grace deadline. Later sessions get the remainder.
    pub async fn shutdown(&self) -> Result<(), Error> {
        let deadline = Instant::now() + self.grace;
        let clients = {
            let servers = lock(&self.servers);
            servers
                .values()
                .map(|slot| slot.client.clone())
                .collect::<Vec<_>>()
        };
        let mut first_error = None;
        for client in clients {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if let Err(err) = client.shutdown_within(remaining).await {
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
        // Status is down so a late call does not use a closed session.
        let mut servers = lock(&self.servers);
        for slot in servers.values_mut() {
            slot.status = Status::Down;
            self.forget_installed(slot);
        }
        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    pub async fn call(
        &self,
        server: &str,
        tool: &str,
        arguments: Value,
        cancel: CancellationToken,
    ) -> Result<ToolOutput, Error> {
        let namespaced = format!("{server}__{tool}");
        let client = {
            let servers = lock(&self.servers);
            let Some(slot) = servers.get(server) else {
                return Err(tool_err(
                    &namespaced,
                    ToolErrorKind::NotFound,
                    format!("no such tool: {namespaced}"),
                ));
            };
            if slot.status != Status::Up {
                return Err(tool_err(
                    &namespaced,
                    ToolErrorKind::McpDown,
                    "mcp server down",
                ));
            }
            slot.client.clone()
        };
        match client.call(tool, arguments, cancel).await {
            Err(Error::Mcp { .. }) => {
                self.mark_down(server).await;
                Err(tool_err(
                    &namespaced,
                    ToolErrorKind::McpDown,
                    "mcp server down",
                ))
            }
            other => other,
        }
    }

    pub async fn pid(&self, server: &str) -> Option<u32> {
        let client = {
            let servers = lock(&self.servers);
            servers.get(server).map(|slot| slot.client.clone())
        }?;
        client.pid().await
    }

    pub fn tools_skipped(&self, server: &str) -> u32 {
        lock(&self.servers)
            .get(server)
            .map(|slot| slot.tools_skipped)
            .unwrap_or(0)
    }

    /// Names whose session status is down. This is the only status query.
    pub fn down_servers(&self) -> Vec<String> {
        lock(&self.servers)
            .iter()
            .filter(|(_, slot)| slot.status == Status::Down)
            .map(|(name, _)| name.clone())
            .collect()
    }

    async fn poll_server(&self, name: &str) -> Result<(), Error> {
        let (up, waiting, client) = {
            let servers = lock(&self.servers);
            let Some(slot) = servers.get(name) else {
                return Ok(());
            };
            let waiting = slot.next_attempt.is_some_and(|at| at > Instant::now());
            (slot.status == Status::Up, waiting, slot.client.clone())
        };
        if up {
            if client.transport_dead().await {
                // Do not arm backoff here. This poll still gets one connect attempt.
                self.mark_down(name).await;
                return self.try_connect(name).await;
            }
            if client.take_list_changed() {
                return self.relist(name, client).await;
            }
            return Ok(());
        }
        if waiting {
            return Ok(());
        }
        self.try_connect(name).await
    }

    async fn relist(&self, name: &str, client: StdioMcpClient) -> Result<(), Error> {
        match client.list_tools().await {
            Ok(defs) => {
                self.install(name, defs);
                Ok(())
            }
            Err(Error::Cancelled) => Err(Error::Cancelled),
            Err(_) => {
                self.mark_down(name).await;
                self.try_connect(name).await
            }
        }
    }

    async fn try_connect(&self, name: &str) -> Result<(), Error> {
        let (client, required) = {
            let servers = lock(&self.servers);
            let Some(slot) = servers.get(name) else {
                return Ok(());
            };
            (slot.client.clone(), slot.required)
        };
        match client.connect().await {
            Ok(()) => match client.list_tools().await {
                Ok(defs) => {
                    self.install(name, defs);
                    self.mark_up(name);
                    Ok(())
                }
                Err(Error::Cancelled) => Err(Error::Cancelled),
                Err(err) => self.fail_connect(name, &err.to_string(), required).await,
            },
            Err(Error::Cancelled) => Err(Error::Cancelled),
            Err(err) => self.fail_connect(name, &err.to_string(), required).await,
        }
    }

    /// Failed connect attempt. Backoff advances only here.
    async fn fail_connect(&self, name: &str, message: &str, required: bool) -> Result<(), Error> {
        let message = redact(message);
        let client = {
            let mut servers = lock(&self.servers);
            let Some(slot) = servers.get_mut(name) else {
                return Ok(());
            };
            slot.status = Status::Down;
            let wait = slot.backoff;
            slot.next_attempt = Some(Instant::now() + wait);
            slot.backoff = (slot.backoff * 2).min(BACKOFF_CAP);
            self.forget_installed(slot);
            slot.client.clone()
        };
        let _ = client.shutdown_within(self.grace).await;
        if required {
            Err(Error::Mcp {
                server: name.to_string(),
                message,
            })
        } else {
            Ok(())
        }
    }

    async fn mark_down(&self, name: &str) {
        let client = {
            let mut servers = lock(&self.servers);
            let Some(slot) = servers.get_mut(name) else {
                return;
            };
            slot.status = Status::Down;
            self.forget_installed(slot);
            slot.client.clone()
        };
        let _ = client.shutdown_within(self.grace).await;
    }

    fn mark_up(&self, name: &str) {
        let mut servers = lock(&self.servers);
        let Some(slot) = servers.get_mut(name) else {
            return;
        };
        slot.status = Status::Up;
        slot.backoff = BACKOFF_START;
        slot.next_attempt = None;
        let down = std::mem::take(&mut slot.down_names);
        self.registry
            .clear_mcp_down(down.iter().map(String::as_str));
    }

    /// Drop only the names this server installed. A colliding `{server}__*` tool
    /// this server skipped stays registered.
    fn forget_installed(&self, slot: &mut ServerSlot) {
        for name in std::mem::take(&mut slot.registered) {
            self.registry.mark_mcp_down(&name);
            self.registry.unregister(&name);
            slot.down_names.insert(name);
        }
    }

    fn install(&self, name: &str, defs: Vec<ToolDefinition>) {
        let weak = self.weak.clone();
        let server = name.to_string();
        let mut servers = lock(&self.servers);
        let Some(slot) = servers.get_mut(name) else {
            return;
        };
        let outcome = install_tools(&self.registry, name, defs, &slot.registered, |prepared| {
            mcp_tool(weak.clone(), &server, prepared)
        });
        slot.registered = outcome.registered;
        slot.tools_skipped = outcome.tools_skipped;
    }
}

fn mcp_tool(
    supervisor: Weak<McpSupervisor>,
    server: &str,
    prepared: PreparedTool,
) -> Arc<dyn Tool> {
    Arc::new(McpTool {
        supervisor,
        server: server.to_string(),
        tool: prepared.server_name,
        namespaced: prepared.namespaced,
        description: prepared.description,
        parameters: prepared.parameters,
    })
}

struct McpTool {
    supervisor: Weak<McpSupervisor>,
    server: String,
    tool: String,
    namespaced: String,
    description: String,
    parameters: Value,
}

#[async_trait::async_trait]
impl Tool for McpTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.namespaced.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }

    async fn call(&self, ctx: ToolCtx, arguments: Value) -> Result<ToolOutput, Error> {
        let Some(supervisor) = self.supervisor.upgrade() else {
            return Err(tool_err(
                &self.namespaced,
                ToolErrorKind::McpDown,
                "mcp server down",
            ));
        };
        supervisor
            .call(&self.server, &self.tool, arguments, ctx.cancel)
            .await
    }
}

#[async_trait::async_trait]
impl SessionPoll for McpSupervisor {
    async fn poll(&self) -> Result<(), Error> {
        let _guard = self.poll_lock.lock().await;
        let first = self.first_poll.swap(false, Ordering::AcqRel);
        let names: Vec<String> = lock(&self.servers).keys().cloned().collect();
        let mut required_error = None;
        for name in names {
            if let Err(err) = self.poll_server(&name).await {
                if first && required_error.is_none() {
                    required_error = Some(err);
                }
            }
        }
        match required_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

fn tool_err(tool: &str, kind: ToolErrorKind, message: impl Into<String>) -> Error {
    Error::Tool {
        tool: tool.to_string(),
        kind,
        message: message.into(),
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}
