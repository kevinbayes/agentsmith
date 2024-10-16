use std::sync::{Arc, Mutex};
use std::time::Duration;
use futures_util::TryFutureExt;
use reqwest::{Proxy, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use agentsmith_common::config::config::Config;
use crate::llm::llm::{GenerateText, LLMResult, Prompt};
use agentsmith_common::error::error::{Error, Result};
use chrono::Local;
use tracing::info;
use tracing_subscriber;

#[derive(Clone, Debug)]
pub struct AnthropicLLM {
    api_key: String,
    base_url: String,
    model: String,
    client: Arc<Mutex<reqwest::Client>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: Role,
    pub content: Vec<AnthropicMessageContent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    Text { type_: String, text: String },
    Image { type_: String, source: AnthropicMessageContentSource },
    ToolUse { type_: String, id: String, name: String, input: Option<Value> },
    ToolResult { type_: String, tool_use_id: String, is_error: Option<bool>, content: Option<Value> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text { type_: String, text: String },
    Image { type_: String, source: AnthropicMessageContentSource },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicMessageContentSource {
    pub type_ : String,
    pub media_type: String,
    pub data: String
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicMetadata {
    pub user_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Role {
    #[serde(rename = "human")]
    Human,
    #[serde(rename = "assistant")]
    Assistant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Auto { type_: String, disable_parallel_tool_use: Option<bool> },
    Any { type_: String, disable_parallel_tool_use: Option<bool> },
    Tool { type_: String, name: String, disable_parallel_tool_use: Option<bool> },
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
    pub metadata: Option<AnthropicMetadata>,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
    pub tool_choice: Option<ToolChoice>,
    pub tools: Option<Vec<Tool>>
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnthropicGenerateResponse {
    pub content: Vec<Content>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Content {
    pub text: Option<String>,
    pub r#type: String,
    pub id: Option<String>,
    pub name: Option<String>,
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

    pub fn new(config: Config, model: String) -> Self {

        let anthropic_config = config.config.gateways.registry
            .get("anthropic_gateway")
            .unwrap();

        let api_key = anthropic_config.api_key.clone();
        let base_url = anthropic_config.baseurl.clone();
        let client = Arc::new(Mutex::new(reqwest::ClientBuilder::new()
            .connect_timeout(Duration::from_secs(60))
            .build()
            .unwrap()));

        Self { api_key, base_url, model, client }
    }


}

impl GenerateText for AnthropicLLM {

    async fn generate_text(&self, prompt: &Prompt) -> Result<LLMResult> {
        let url_str = format!("{}{}", self.base_url.clone(), "/v1/messages");
        let api_key = self.api_key.clone();
        let model = self.model.clone();

        let request_obj = AnthropicRequest {
            model,
            stream: Some(false),
            tool_choice: None,
            system: prompt.clone().system,
            metadata: None,
            messages: vec![AnthropicMessage {
                role: Role::Human,
                content: vec![AnthropicMessageContent::Text { type_: String::from("text"), text: prompt.clone().system}],
            }],
            temperature: Some(1.0),
            max_tokens: Some(500),
            top_p: Some(1.0),
            stop_sequences: None,
            tools: None,
        };

        let client = self.client.lock().unwrap();

        let request =  client.post(&url_str)
            .bearer_auth(&api_key)
            .header("User-Agent", format!("AgentSmith Framework"))
            .header("Content-Type", format!("application/json"))
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
                AnthropicGenerateResponse{
                    content: vec![],
                    model: "".to_string(),
                    stop_reason: None,
                    usage: Usage { input_tokens: 0, output_tokens: 0 },
                }
            },
        };

        println!("response {:?}", p);
    }
}
