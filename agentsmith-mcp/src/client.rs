use agentsmith_core::Error;
use agentsmith_tools::{ToolDefinition, ToolOutput};
use tokio_util::sync::CancellationToken;

/// v1 has one implementer, [`crate::StdioMcpClient`].
#[async_trait::async_trait]
pub trait McpClient: Send + Sync {
    async fn connect(&self) -> Result<McpSessionInfo, Error>;
    async fn list_tools(&self) -> Result<Vec<ToolDefinition>, Error>;
    async fn call(
        &self,
        name: &str,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> Result<ToolOutput, Error>;
    async fn shutdown(&self) -> Result<(), Error>;
}

pub struct McpSessionInfo {
    pub server: String,
    pub server_version: Option<String>,
    pub tool_count: u32,
}
