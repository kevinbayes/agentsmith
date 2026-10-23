use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agentsmith_core::config::McpServerConfig;
use agentsmith_core::{Error, ToolErrorKind};
use agentsmith_tools::{cap_tool_output, ToolDefinition, ToolOutput};
use rmcp::model::{CallToolRequestParams, CallToolResponse, ContentBlock};
use rmcp::service::{NotificationContext, RunningService};
use rmcp::{ClientHandler, RoleClient, ServiceError, ServiceExt};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::client::{McpClient, McpSessionInfo};

const FOLLOW_UP: &str = "mcp follow-up not supported";

struct ToolListWatch {
    changed: Arc<AtomicBool>,
}

impl ClientHandler for ToolListWatch {
    async fn on_tool_list_changed(&self, _context: NotificationContext<RoleClient>) {
        self.changed.store(true, Ordering::Release);
    }
}

type Session = RunningService<RoleClient, ToolListWatch>;

struct Inner {
    service: Option<Session>,
    pid: Option<u32>,
}

struct Shared {
    server: String,
    command: String,
    args: Vec<String>,
    env: BTreeMap<String, String>,
    connect_timeout: Duration,
    call_timeout: Duration,
    max_output_bytes: u64,
    grace: Duration,
    cancel: CancellationToken,
    inner: tokio::sync::Mutex<Inner>,
    changed: Mutex<Arc<AtomicBool>>,
}

/// Child process MCP session. Holds `RunningService` only: `serve` consumes
/// the `TokioChildProcess`, and the pid is the value `id()` returned before that.
#[derive(Clone)]
pub struct StdioMcpClient {
    shared: Arc<Shared>,
}

impl StdioMcpClient {
    pub fn new(
        server: &str,
        config: &McpServerConfig,
        max_output_bytes: u64,
        grace: Duration,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            shared: Arc::new(Shared {
                server: server.to_string(),
                command: config.command.clone(),
                args: config.args.clone(),
                env: config.env.clone(),
                connect_timeout: Duration::from_secs(config.connect_timeout_secs),
                call_timeout: Duration::from_secs(config.call_timeout_secs),
                max_output_bytes,
                grace,
                cancel,
                inner: tokio::sync::Mutex::new(Inner {
                    service: None,
                    pid: None,
                }),
                changed: Mutex::new(Arc::new(AtomicBool::new(false))),
            }),
        }
    }

    pub async fn pid(&self) -> Option<u32> {
        self.shared.inner.lock().await.pid
    }

    pub fn take_list_changed(&self) -> bool {
        let flag = Arc::clone(&lock(&self.shared.changed));
        flag.swap(false, Ordering::AcqRel)
    }

    /// No session, a closed transport, or a pid that is no longer alive.
    pub async fn transport_dead(&self) -> bool {
        let inner = self.shared.inner.lock().await;
        let Some(service) = inner.service.as_ref() else {
            return true;
        };
        if service.is_transport_closed() {
            return true;
        }
        match inner.pid {
            Some(pid) => !process_alive(pid),
            None => false,
        }
    }

    /// `close_with_timeout` is `TokioChildProcess::graceful_shutdown` (close
    /// stdin, wait, then kill). SIGKILL the recorded pid only when that wait
    /// exceeds `timeout` and returns `Ok(None)`. There is no second kill timer.
    pub async fn shutdown_within(&self, timeout: Duration) -> Result<(), Error> {
        let (mut service, pid) = {
            let mut inner = self.shared.inner.lock().await;
            (inner.service.take(), inner.pid.take())
        };
        let Some(service) = service.as_mut() else {
            return Ok(());
        };
        match service.close_with_timeout(timeout).await {
            Ok(Some(_)) => Ok(()),
            Ok(None) => {
                if let Some(pid) = pid {
                    signal_kill(pid);
                }
                Ok(())
            }
            Err(err) => Err(self.mcp(err.to_string())),
        }
    }

    fn mcp(&self, message: impl Into<String>) -> Error {
        Error::Mcp {
            server: self.shared.server.clone(),
            message: message.into(),
        }
    }

    fn tool(&self, name: &str, kind: ToolErrorKind, message: impl Into<String>) -> Error {
        Error::Tool {
            tool: name.to_string(),
            kind,
            message: message.into(),
        }
    }
}

#[async_trait::async_trait]
impl McpClient for StdioMcpClient {
    async fn connect(&self) -> Result<McpSessionInfo, Error> {
        if self.shared.cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        if self.shared.command.is_empty() {
            return Err(self.mcp("command is empty"));
        }
        // A reconnect must close the previous child before spawning another.
        self.shutdown_within(self.shared.grace).await?;

        let mut command = tokio::process::Command::new(&self.shared.command);
        command.args(&self.shared.args);
        for (key, value) in &self.shared.env {
            command.env(key, value);
        }
        // stderr stays inherited so server logs reach the journal and never the harness.
        let (transport, stderr) = rmcp::transport::TokioChildProcess::builder(command)
            .spawn()
            .map_err(|err| self.mcp(format!("spawn: {err}")))?;
        debug_assert!(stderr.is_none(), "production stderr is inherited");
        drop(stderr);
        let pid = transport.id();
        let changed = Arc::new(AtomicBool::new(false));
        let handler = ToolListWatch {
            changed: Arc::clone(&changed),
        };

        let connect_timeout = self.shared.connect_timeout;
        let served = tokio::select! {
            biased;
            _ = self.shared.cancel.cancelled() => {
                return Err(Error::Cancelled);
            }
            result = tokio::time::timeout(connect_timeout, handler.serve(transport)) => result,
        };
        let service = match served {
            Ok(Ok(service)) => service,
            Ok(Err(err)) => return Err(self.mcp(err.to_string())),
            Err(_) => return Err(self.mcp("connect timed out")),
        };
        let version = service.peer_info().and_then(|info| {
            info.server_info
                .as_ref()
                .map(|server| server.version.clone())
        });
        {
            let mut inner = self.shared.inner.lock().await;
            inner.pid = pid;
            inner.service = Some(service);
        }
        *lock(&self.shared.changed) = changed;
        Ok(McpSessionInfo {
            server: self.shared.server.clone(),
            server_version: version,
            tool_count: 0,
        })
    }

    async fn list_tools(&self) -> Result<Vec<ToolDefinition>, Error> {
        let listed = {
            let inner = self.shared.inner.lock().await;
            let Some(service) = inner.service.as_ref() else {
                return Err(self.mcp("server is down"));
            };
            service.list_all_tools().await
        };
        let tools = listed.map_err(|err| self.mcp(err.to_string()))?;
        let defs = tools
            .iter()
            .map(|tool| ToolDefinition {
                name: tool.name.to_string(),
                description: tool
                    .description
                    .as_ref()
                    .map(|text| text.to_string())
                    .unwrap_or_default(),
                parameters: Value::Object(tool.input_schema.as_ref().clone()),
            })
            .collect();
        Ok(defs)
    }

    async fn call(
        &self,
        name: &str,
        arguments: Value,
        cancel: CancellationToken,
    ) -> Result<ToolOutput, Error> {
        if cancel.is_cancelled() {
            return Err(self.tool(name, ToolErrorKind::Cancelled, "cancelled"));
        }
        if self.transport_dead().await {
            return Err(self.mcp("child exited"));
        }
        let arguments = match arguments {
            Value::Null => None,
            Value::Object(map) => Some(map),
            _ => {
                return Err(self.tool(
                    name,
                    ToolErrorKind::InvalidArgs,
                    "arguments must be a JSON object",
                ))
            }
        };
        let mut params = CallToolRequestParams::new(name.to_string());
        if let Some(map) = arguments {
            params = params.with_arguments(map);
        }

        let call_timeout = self.shared.call_timeout;
        let rpc = async {
            let inner = self.shared.inner.lock().await;
            let Some(service) = inner.service.as_ref() else {
                return Err(self.mcp("server is down"));
            };
            service
                .call_tool_once(params)
                .await
                .map_err(|err| self.classify_call(name, err))
        };
        let response = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                return Err(self.tool(name, ToolErrorKind::Cancelled, "cancelled"));
            }
            result = tokio::time::timeout(call_timeout, rpc) => result,
        };
        let response = match response {
            Ok(result) => result?,
            Err(_) => {
                if self.transport_dead().await {
                    return Err(self.mcp("child exited"));
                }
                return Err(self.tool(name, ToolErrorKind::Timeout, "timed out"));
            }
        };
        self.finish_call(name, response)
    }

    async fn shutdown(&self) -> Result<(), Error> {
        self.shutdown_within(self.shared.grace).await
    }
}

impl StdioMcpClient {
    fn classify_call(&self, name: &str, err: ServiceError) -> Error {
        match err {
            ServiceError::TransportSend(_) | ServiceError::TransportClosed => {
                self.mcp(err.to_string())
            }
            _ => self.tool(name, ToolErrorKind::Execution, err.to_string()),
        }
    }

    fn finish_call(&self, name: &str, response: CallToolResponse) -> Result<ToolOutput, Error> {
        match response {
            CallToolResponse::Complete(result) => {
                let mut text = render_content(&result.content);
                if result.is_error == Some(true) {
                    if text.is_empty() {
                        text = "mcp tool error".to_string();
                    }
                    let text = cap_tool_output(&text, self.shared.max_output_bytes);
                    return Err(self.tool(name, ToolErrorKind::Execution, text));
                }
                Ok(ToolOutput {
                    content: cap_tool_output(&text, self.shared.max_output_bytes),
                })
            }
            CallToolResponse::InputRequired(_) | CallToolResponse::Task(_) => {
                Err(self.tool(name, ToolErrorKind::Execution, FOLLOW_UP))
            }
            _ => Err(self.tool(name, ToolErrorKind::Execution, FOLLOW_UP)),
        }
    }
}

fn render_content(blocks: &[ContentBlock]) -> String {
    let mut out = String::new();
    for block in blocks {
        match block {
            ContentBlock::Text(text) => out.push_str(&text.text),
            ContentBlock::Image(_) => out.push_str("[image omitted]"),
            ContentBlock::Audio(_) => out.push_str("[audio omitted]"),
            ContentBlock::Resource(_) | ContentBlock::ResourceLink(_) => {
                out.push_str("[resource omitted]")
            }
            _ => out.push_str("[content omitted]"),
        }
    }
    out
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}

/// `kill(pid, 0)` checks liveness. pid 0 would signal the whole process group.
fn process_alive(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    let rc = unsafe { libc::kill(pid, 0) };
    rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

fn signal_kill(pid: u32) {
    let Ok(pid) = i32::try_from(pid) else {
        return;
    };
    if pid <= 0 {
        return;
    }
    unsafe {
        libc::kill(pid, libc::SIGKILL);
    }
}
