use std::sync::{Arc, Mutex};
use std::time::Duration;
use futures_util::TryFutureExt;
use reqwest::{Proxy, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use agentsmith_common::config::config::{Config, GatewayConfig};
use crate::llm::llm::{GenerateText, LLMConfiguration, LLMResult};
use agentsmith_common::error::error::{Error, Result};
use chrono::Local;
use tracing::info;
use tracing_subscriber;
use crate::llm::prompt::{AssistantContent,UserContent, Prompt, PromptMessage};

#[derive(Clone, Debug)]
pub struct AnthropicLLM {
    global_config: GatewayConfig,
    config: LLMConfiguration,
    client: Arc<Mutex<reqwest::Client>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: Role,
    pub content: Vec<AnthropicMessageContent>,
}

impl AnthropicMessage {

    pub fn from_prompt_message(prompt: &PromptMessage) -> Self {
        let prompt = prompt.clone();
        match prompt {
            PromptMessage::User { role, content, name } => {

                let content = content.iter().map(AnthropicMessageContent::map_from_content).collect();
                Self { role: Role::User, content, }
            }
            PromptMessage::Assistant { role, content, refusal, name, tool_calls } => {
                let content = content.unwrap().iter().map(AnthropicMessageContent::map_assistant_content).collect();
                Self { role: Role::Assistant, content, }
            }
            _ => {
                Self { role: Role::User, content: vec![], }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    Text {
        #[serde(rename = "type")]
        type_: String,
        text: String,
    },
    Image {
        #[serde(rename = "type")]
        type_: String,
        source: AnthropicMessageContentSource,
    },
    ToolUse {
        #[serde(rename = "type")]
        type_: String,
        id: String,
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        input: Option<Value>,
    },
    ToolResult {
        #[serde(rename = "type")]
        type_: String,
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<Value>,
    },
}

impl AnthropicMessageContent {
    fn map_from_content(content: &UserContent) -> Self {
        match content {
            UserContent::Text { text, type_} => {
                Self::Text {type_: type_.clone(), text: text.clone()}
            }
            UserContent::Image { content_type, image_url, type_ } => {
                let image_url = image_url.clone();
                Self::Image { type_: type_.clone(), source: AnthropicMessageContentSource {
                    type_: "base64".to_string(),
                    media_type: content_type.clone().unwrap_or("image/png".to_string()),
                    data: image_url.url
                } }
            }
        }
    }
    fn map_assistant_content(content: &AssistantContent) -> Self {
        match content {
            AssistantContent::Text { text, type_ } => {
                Self::Text {type_: type_.clone(), text: text.clone()}
            }
            AssistantContent::Image { content_type, image_url, type_ } => {
                let image_url = image_url.clone();
                Self::Image { type_: type_.clone(), source: AnthropicMessageContentSource {
                    type_: "base64".to_string(),
                    media_type: content_type.clone().unwrap_or("image/png".to_string()),
                    data: image_url.url
                } }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text {
        #[serde(rename = "type")]
        type_: String,
        text: String,
    },
    Image {
        #[serde(rename = "type")]
        type_: String,
        source: AnthropicMessageContentSource,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicMessageContentSource {
    #[serde(rename = "type")]
    pub type_: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicMetadata {
    pub user_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Role {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Auto {
        #[serde(rename = "type")]
        type_: String,
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        #[serde(rename = "type")]
        type_: String,
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        #[serde(rename = "type")]
        type_: String,
        name: String,
        disable_parallel_tool_use: Option<bool>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub system: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AnthropicMetadata>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicConfig {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AnthropicMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

impl AnthropicRequest {

    fn from_prompt(config: &LLMConfiguration, prompt: &Prompt) -> Self {

        let config = config.clone();

        match prompt {
            Prompt::Simple { system, user, tools, tool_choice } => {

                let system = system.clone();
                let user = user.clone();

                AnthropicRequest {
                    model: config.model,
                    stream: Some(false),
                    tool_choice: None,
                    system: system,
                    metadata: None,
                    messages: vec![AnthropicMessage {
                        role: Role::User,
                        content: vec![AnthropicMessageContent::Text {
                            type_: String::from("text"),
                            text: user
                        }],
                    }],
                    temperature: config.temperature,
                    max_tokens: config.max_tokens,
                    top_p: None,
                    stop_sequences: None,
                    tools: None,
                }
            }
            Prompt::Messages { system, messages, tools, tool_choice } => {

                let system = system.clone();
                let messages = messages.clone();

                let request_messages: Vec<AnthropicMessage> = messages.iter().map(AnthropicMessage::from_prompt_message).collect();

                AnthropicRequest {
                    model: config.model,
                    stream: Some(false),
                    tool_choice: None,
                    system: system,
                    metadata: None,
                    messages: request_messages,
                    temperature: Some(1.0),
                    max_tokens: Some(500),
                    top_p: Some(1.0),
                    stop_sequences: None,
                    tools: None,
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicGenerateResponse {
    pub content: Vec<Content>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicError {
    pub r#type: String,
    pub message: String,
}


impl AnthropicLLM {
    pub fn new(config: Config, llm_configuration: LLMConfiguration) -> Self {

        let global_config = config.config.gateways.registry
            .get("anthropic_gateway")
            .unwrap()
            .clone();

        let client = Arc::new(Mutex::new(reqwest::ClientBuilder::new()
            .connect_timeout(Duration::from_secs(60))
            .build()
            .unwrap()));

        Self { global_config: global_config, config: llm_configuration, client }
    }
}

impl GenerateText for AnthropicLLM {
    async fn generate(&self, prompt: &Prompt) -> Result<LLMResult> {

        let global_config = self.global_config.clone();
        let config = self.config.clone();

        let url_str = format!("{}{}", config.base_url.unwrap_or(global_config.baseurl), "/v1/messages");
        let api_key = config.credentials.api_key.clone();
        let api_version = config.version.unwrap_or("2023-06-01".to_string());

        let request_obj = AnthropicRequest::from_prompt(&self.config, prompt);

        let client = self.client.lock().unwrap();

        let request = client.post(&url_str)
            .header("User-Agent", "AgentSmith Framework".to_string())
            .header("anthropic-version", api_version)
            .header("x-api-key", format!("{}", api_key))
            .header("Content-Type", "application/json".to_string())
            .json(&request_obj)
            .build()
            .map_err(|e| {
                println!("Error: {:?}", e);
                Error::AgentError { id: 0, code: 2 }
            })?;
        info!("Request: {:?}", request);
        info!("Request Body: {:?}", request_obj);

        let res = client.execute(request)
            .map_err(|e| {
                println!("Error: {:?}", e);
                Error::AgentError { id: 0, code: 2 }
            })
            .await?
            .json::<AnthropicGenerateResponse>()
            .map_err(|e| {
                println!("Error: {:?}", e);
                Error::AgentError { id: 0, code: 2 }
            })
            .await?;
        //
        Ok(LLMResult::from_anthropic(&res))
    }
}


#[cfg(test)]
mod tests {
    use std::fs;
    use agentsmith_common::config::config::read_config;
    use super::*;


    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_parse() {
        let data = r#"{
  "content": [
    {
      "text": "Hi! My name is Claude.",
      "type": "text"
    }
  ],
  "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
  "model": "claude-3-5-sonnet-20240620",
  "role": "assistant",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "type": "message",
  "usage": {
    "input_tokens": 2095,
    "output_tokens": 503
  }
}"#;

        let now = Local::now().timestamp_millis();

        // Parse the string of data into serde_json::Value.
        let p: AnthropicGenerateResponse = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                println!("Error: {:?}", e);
                AnthropicGenerateResponse {
                    content: vec![],
                    model: "".to_string(),
                    stop_reason: None,
                    usage: Usage { input_tokens: 0, output_tokens: 0 },
                }
            }
        };

        println!("response {:?}", p);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_tools_parse() {
        let data = r#"{
  "id": "msg_01RRSnhxG1w1DVsvQA2qm2Mm",
  "type": "message",
  "role": "assistant",
  "model": "claude-3-5-sonnet-20240620",
  "content": [
    {
      "type": "text",
      "text": "Certainly! I can help you with that information. To get the current weather in San Francisco, I'll use the get_weather function. Let me fetch that data for you."
    },
    {
      "type": "tool_use",
      "id": "toolu_012UQk4kPcYF7XYj67iZzbwi",
      "name": "get_weather",
      "input": {
        "location": "San Francisco, CA"
      }
    }
  ],
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 384,
    "output_tokens": 94
  }
}"#;

        let now = Local::now().timestamp_millis();

        // Parse the string of data into serde_json::Value.
        let p: AnthropicGenerateResponse = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                println!("Error: {:?}", e);
                AnthropicGenerateResponse {
                    content: vec![],
                    model: "".to_string(),
                    stop_reason: None,
                    usage: Usage { input_tokens: 0, output_tokens: 0 },
                }
            }
        };

        println!("response {:?}", p);
    }
}
