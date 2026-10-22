use std::path::PathBuf;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use agentsmith_core::{Error, ToolErrorKind};

/// JSON schema and description sent to the model for one tool.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Per-call context. `cancel` is checked before a builtin starts work.
#[derive(Clone, Debug)]
pub struct ToolCtx {
    pub run_id: uuid::Uuid,
    pub workdir: PathBuf,
    pub cancel: CancellationToken,
}

/// Successful tool result. The caller caps what the model sees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolOutput {
    pub content: String,
}

/// Object-safe tool. Native `async fn` in traits is not dyn-compatible.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    async fn call(&self, ctx: ToolCtx, arguments: Value) -> Result<ToolOutput, Error>;
}

pub(crate) fn tool_error(tool: &str, kind: ToolErrorKind, message: impl Into<String>) -> Error {
    Error::Tool {
        tool: tool.to_string(),
        kind,
        message: message.into(),
    }
}

pub(crate) fn ensure_not_cancelled(tool: &str, ctx: &ToolCtx) -> Result<(), Error> {
    if ctx.cancel.is_cancelled() {
        Err(tool_error(tool, ToolErrorKind::Cancelled, "cancelled"))
    } else {
        Ok(())
    }
}
