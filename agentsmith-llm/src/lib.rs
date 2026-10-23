//! Chat clients. `llm.provider` selects one at process start.

mod anthropic;
mod arguments;
mod gemini;
mod http;
mod openai;
mod types;

pub use anthropic::Anthropic;
pub use gemini::Gemini;
pub use openai::OpenAiCompatible;
pub use types::{
    Completion, CompletionRequest, Llm, Message, StopReason, ToolCall, ToolChoice, ToolDefinition,
};

use std::time::Duration;

use agentsmith_core::config::LlmProvider;
use agentsmith_core::Error;

/// The client selected by deploy config.
pub enum LlmClient {
    OpenAi(OpenAiCompatible),
    Anthropic(Anthropic),
    Gemini(Gemini),
}

impl std::fmt::Debug for LlmClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAi(client) => client.fmt(f),
            Self::Anthropic(client) => client.fmt(f),
            Self::Gemini(client) => client.fmt(f),
        }
    }
}

/// Build the client for `provider`. The loop does not know which one it is.
pub fn connect(
    provider: LlmProvider,
    base_url: &str,
    api_key: impl Into<String>,
    timeout: Duration,
    retries: u32,
) -> Result<LlmClient, Error> {
    let api_key = api_key.into();
    Ok(match provider {
        LlmProvider::OpenAi => {
            LlmClient::OpenAi(OpenAiCompatible::new(base_url, api_key, timeout, retries)?)
        }
        LlmProvider::Anthropic => {
            LlmClient::Anthropic(Anthropic::new(base_url, api_key, timeout, retries)?)
        }
        LlmProvider::Gemini => LlmClient::Gemini(Gemini::new(base_url, api_key, timeout, retries)?),
    })
}

#[async_trait::async_trait]
impl Llm for LlmClient {
    async fn complete(&self, request: CompletionRequest) -> Result<Completion, Error> {
        match self {
            Self::OpenAi(client) => client.complete(request).await,
            Self::Anthropic(client) => client.complete(request).await,
            Self::Gemini(client) => client.complete(request).await,
        }
    }
}
