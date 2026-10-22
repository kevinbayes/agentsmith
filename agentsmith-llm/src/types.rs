use agentsmith_core::Error;
use tokio_util::sync::CancellationToken;

/// Chat-completions client.
///
/// `async_trait` keeps this dyn-compatible so another provider can be added
/// without changing the loop. v1 ships only [`crate::OpenAiCompatible`].
#[async_trait::async_trait]
pub trait Llm: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<Completion, Error>;
}

#[derive(Debug)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub max_output_tokens: u32,
    pub temperature: Option<f32>,
    pub cancel: CancellationToken,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    /// Planner forces the single tool `submit_plan`.
    Function {
        name: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Wire text. A JSON string is kept as that string. A JSON object is stored
    /// as its compact JSON text.
    pub arguments_raw: String,
    /// `None` when `arguments_raw` is not JSON. The loop records `invalid_args`
    /// and does not call the tool. [`Llm::complete`] still returns `Ok`.
    pub arguments: Option<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, PartialEq)]
pub struct Completion {
    /// Always [`Message::Assistant`].
    pub message: Message,
    pub stop: StopReason,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    Length,
    ContentFilter,
}
