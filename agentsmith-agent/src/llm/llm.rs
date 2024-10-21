use log::{debug, info};
use crate::llm::anthropic_llm::AnthropicGenerateResponse;
use crate::llm::openai_llm::OpenAIGenerateResponse;
use crate::llm::prompt::Prompt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LLMConfiguration {
    #[serde(rename = "credentials")]
    pub credentials: LLMCredentials,
    #[serde(rename = "model")]
    pub base_url: Option<String>,
    #[serde(rename = "version")]
    pub version: Option<String>,
    #[serde(rename = "model")]
    pub model: String,
    #[serde(rename = "stream")]
    pub stream: Option<bool>,
    #[serde(rename = "temperature")]
    pub temperature: Option<f32>,
    #[serde(rename = "max_tokens")]
    pub max_tokens: Option<i32>,
    #[serde(rename = "seed")]
    pub seed: Option<i32>,
    #[serde(rename = "top_p")]
    pub top_p: Option<i32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LLMCredentials {
    pub api_key: String,
}

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
    pub name: String,
    pub input: Option<Value>
}

impl LLMResultToolCall {

    fn from_openai(tool_calls: Option<Vec<Value>>) -> Option<Vec<LLMResultToolCall>> {
        match tool_calls {
            Some(tool_calls) => {
                Some(tool_calls.iter().map(|item| {
                    let item = item.clone();

                    let id = match item.get("id") {
                        Some(Value::String(id)) => id.clone(),
                        _ => "".to_string(),
                    };

                    let type_ = match item.get("type") {
                        Some(Value::String(type_)) => type_.clone(),
                        _ => "function".to_string(),
                    };

                    let function_value = match item.get("function") {
                        Some(function) => function.clone(),
                        _ => Value::Null
                    };

                    let name_and_argument = match function_value {
                        Value::Object(o) => {
                            (o.get("name").map(|x| x.clone()), o.get("arguments").map(|x| x.clone()))
                        }
                        _ => (None, None)
                    };

                    (id, type_, name_and_argument.0, name_and_argument.1)
                }).filter(|o| o.2.is_some() && o.3.is_some())
                    .map(|o| Self {
                        id: o.0,
                        type_: o.1,
                        name: String::from(o.2.unwrap().as_str().unwrap()),
                        input: serde_json::from_str(o.3.unwrap().as_str().unwrap()).unwrap(),
                    })
                    .collect())
            }
            None => None,
        }
    }
}

impl LLMResult {

    pub fn new(message: String) -> Self {
        Self { message, result: "text".to_string(), tool_calls: vec![] }
    }

    pub fn from_cerebras(cerebras_response: &OpenAIGenerateResponse) -> Self {

        let response = cerebras_response.clone();
        let choice = if response.choices.len() > 0 {
            Some(response.choices[0].clone())
        } else {
            None
        };

        match choice {
            Some(choice) => {
                info!("Creating response from {:?}", choice.clone());
                let tool_calls = LLMResultToolCall::from_openai(choice.message.clone().tool_calls)
                    .unwrap_or(Vec::new());

                debug!("Creating response tools {:?}", tool_calls.clone());
                Self { message: choice.message.content.unwrap_or("".to_string()), result: "".to_string(), tool_calls }
            },
            None => {
                Self { message: "No response received.".to_string(), result: "error".to_string(), tool_calls: vec![] }
            }
        }
    }


    pub fn from_openai(cerebras_response: &OpenAIGenerateResponse) -> Self {

        let response = cerebras_response.clone();
        let choice = if response.choices.len() > 0 {
            Some(response.choices[0].clone())
        } else {
            None
        };

        match choice {
            Some(choice) => {
                info!("Creating response from {:?}", choice.clone());
                let tool_calls = LLMResultToolCall::from_openai(choice.message.clone().tool_calls)
                    .unwrap_or(Vec::new());

                debug!("Creating response tools {:?}", tool_calls.clone());
                Self { message: choice.message.content.unwrap_or("".to_string()), result: "".to_string(), tool_calls }
            },
            None => {
                Self { message: "No response received.".to_string(), result: "error".to_string(), tool_calls: vec![] }
            }
        }
    }

    pub fn from_groq(cerebras_response: &OpenAIGenerateResponse) -> Self {

        let response = cerebras_response.clone();
        let choice = if response.choices.len() > 0 {
            Some(response.choices[0].clone())
        } else {
            None
        };

        match choice {
            Some(choice) => {
                info!("Creating response from {:?}", choice.clone());
                let tool_calls = LLMResultToolCall::from_openai(choice.message.clone().tool_calls)
                    .unwrap_or(Vec::new());

                debug!("Creating response tools {:?}", tool_calls.clone());
                Self { message: choice.message.content.unwrap_or("".to_string()), result: "".to_string(), tool_calls }
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
