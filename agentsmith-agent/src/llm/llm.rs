use serde::{Deserialize, Serialize};
use serde_json::Value;
use crate::llm::anthropic_llm::AnthropicGenerateResponse;
use crate::llm::openai_llm::{AssistantContent, AssistantToolCall, OpenAIGenerateResponse, UserContent};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LLMResult {
    pub message: String,
    pub result: String,
    pub tool_calls: Vec<LLMResultToolCall>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LLMResultToolCall {
    pub id: String,
    pub type_: String,
    pub function: Option<Value>
}

impl LLMResult {

    pub fn new(message: String) -> Self {
        Self { message, result: "text".to_string(), tool_calls: vec![] }
    }

    pub fn from_cerebras(cerebras_response: &OpenAIGenerateResponse) -> Self {

        let response = cerebras_response.clone();
        let choice = if response.choices.len() == 0 {
            Some(response.choices[0].clone())
        } else {
            None
        };

        match choice {
            Some(choice) => {
                Self { message: choice.message.content.unwrap(), result: "error".to_string(), tool_calls: vec![] }
            },
            None => {
                Self { message: "No response received.".to_string(), result: "error".to_string(), tool_calls: vec![] }
            }
        }
    }


    pub fn from_openai(cerebras_response: &OpenAIGenerateResponse) -> Self {

        let response = cerebras_response.clone();
        let choice = if response.choices.len() == 0 {
            Some(response.choices[0].clone())
        } else {
            None
        };

        match choice {
            Some(choice) => {
                Self { message: choice.message.content.unwrap(), result: "error".to_string(), tool_calls: vec![] }
            },
            None => {
                Self { message: "No response received.".to_string(), result: "error".to_string(), tool_calls: vec![] }
            }
        }
    }

    pub fn from_groq(cerebras_response: &OpenAIGenerateResponse) -> Self {

        let response = cerebras_response.clone();
        let choice = if response.choices.len() == 0 {
            Some(response.choices[0].clone())
        } else {
            None
        };

        match choice {
            Some(choice) => {
                Self { message: choice.message.content.unwrap(), result: "error".to_string(), tool_calls: vec![] }
            },
            None => {
                Self { message: "No response received.".to_string(), result: "error".to_string(), tool_calls: vec![] }
            }
        }
    }

    pub fn from_anthropic(anthropic_generate_response: &AnthropicGenerateResponse) -> Self {

        let response = anthropic_generate_response.clone();
        let stop_reason = response.stop_reason;
        let content = response.content[0].clone();

        match stop_reason {
            Some(stop_reason) => {
                match stop_reason.as_str() {
                    "end_turn" => {
                        Self { message: content.text.unwrap(), result: "".to_string(), tool_calls: vec![] }
                    },
                    "max_tokens" => {
                        Self { message: content.text.unwrap(), result: "".to_string(), tool_calls: vec![] }
                    },
                    "stop_sequence" => {
                        Self { message: content.text.unwrap(), result: "".to_string(), tool_calls: vec![] }
                    },
                    "tool_use" => {
                        Self { message: content.text.unwrap(), result: "".to_string(), tool_calls: vec![] }
                    },
                    _ => {
                        Self { message: content.text.unwrap(), result: "".to_string(), tool_calls: vec![] }
                    }
                }
            },
            None => {
                Self { message: content.text.unwrap(), result: "".to_string(), tool_calls: vec![] }
            }
        }
    }
}

pub trait GenerateText {

    async fn generate(&self, prompt: &Prompt) -> agentsmith_common::error::error::Result<LLMResult>;
}
