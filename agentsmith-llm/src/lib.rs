//! OpenAI-compatible chat completions.

mod openai;
mod types;

pub use openai::OpenAiCompatible;
pub use types::{
    Completion, CompletionRequest, Llm, Message, StopReason, ToolCall, ToolChoice, ToolDefinition,
};
