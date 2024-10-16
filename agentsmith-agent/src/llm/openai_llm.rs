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
pub struct OpenAILLM {
    api_key: String,
    base_url: String,
    model: String,
    client: Arc<Mutex<reqwest::Client>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIRequest {
    #[serde(rename = "model")]
    pub model: String,
    #[serde(rename = "stream")]
    pub stream: bool,
    #[serde(rename = "messages")]
    pub messages: Vec<OpenAIRequestMessage>,
    #[serde(rename = "temperature")]
    pub temperature: f32,
    #[serde(rename = "max_tokens")]
    pub max_tokens: i32,
    #[serde(rename = "seed")]
    pub seed: i32,
    #[serde(rename = "top_p")]
    pub top_p: i32,
    pub tool_choice: Option<ToolChoice>,
    pub tools: Option<Vec<Tool>>
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIRequestMessage {
    System { role: String, content: String, name: Option<String> },
    User { role: String, content: Vec<UserContent>, name: Option<String> },
    Assistant { role: String, content: Option<Vec<AssistantContent>>, refusal: Option<String>, name: Option<String>, tool_calls: Option<AssistantToolCall> },
    Tool { role: String, content: Vec<String>, name: String },
}


#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text { type_: String, text: String },
    Image { type_: String, image_url: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AssistantContent {
    Text { type_: String, text: String },
    Image { type_: String, refusal: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssistantToolCall {
    pub id: String,
    pub type_: String,
    pub function: Value,
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

#[derive(Clone, Debug,Deserialize, Serialize)]
pub struct OpenAIGenerateResponse {
    pub id: String,
    pub choices: Vec<OpenAIGenerateResponseChoice>,
    pub created: i64,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: OpenAIGenerateResponseUsage,
    pub time_info: Option<OpenAIGenerateResponseTimeInfo>,
}

#[derive(Clone, Debug,Deserialize, Serialize)]
pub struct OpenAIGenerateResponseChoice {
    pub finish_reason: String,
    pub index: u32,
    pub message: OpenAIGenerateResponseMessage,
}

#[derive(Clone, Debug,Deserialize, Serialize)]
pub struct OpenAIGenerateResponseMessage {
    pub content: Option<String>,
    pub role: String,
    pub tool_calls: Option<Value>,
}

#[derive(Clone, Debug,Deserialize, Serialize)]
pub struct OpenAIGenerateResponseUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Clone, Debug,Deserialize, Serialize)]
pub struct OpenAIGenerateResponseTimeInfo {
    pub queue_time: f64,
    pub prompt_time: f64,
    pub completion_time: f64,
    pub total_time: f64,
    pub created: i64,
}


impl OpenAILLM {

    pub fn new(config: Config, model: String) -> Self {

        let openai_config = config.config.gateways.registry
            .get("openai_gateway")
            .unwrap();

        let api_key = openai_config.api_key.clone();
        let base_url = openai_config.baseurl.clone();
        let client = Arc::new(Mutex::new(reqwest::ClientBuilder::new()
            .connect_timeout(Duration::from_secs(60))
            .build()
            .unwrap()));

        Self { api_key, base_url, model, client }
    }


}

impl GenerateText for OpenAILLM {

    async fn generate_text(&self, prompt: &Prompt) -> Result<LLMResult> {
        let url_str = format!("{}{}", self.base_url.clone(), "/v1/chat/completions");
        let api_key = self.api_key.clone();
        let model = self.model.clone();

        let request_obj = OpenAIRequest {
            model,
            stream: false,
            messages: vec![OpenAIRequestMessage::System {
                role: String::from("system"),
                content: prompt.clone().system,
                name: None,
            }, OpenAIRequestMessage::User {
                role: String::from("user"),
                content: vec![UserContent::Text { type_: "text".to_string(), text: prompt.clone().user }],
                name: None,
            }],
            temperature: 1.0,
            max_tokens: 500,
            seed: 0,
            top_p: 1,
            tool_choice: None,
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
            .json::<OpenAIGenerateResponse>()
            .map_err(|e| {
                println!("Error: {:?}", e);
                Error::AgentError { id: 0, code: 2 }
            })
            .await?;

        Ok(LLMResult::from_openai(&res))
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
  "id": "chatcmpl-292e278f-514e-4186-9010-91ce6a14168b",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Hello! How can I assist you today?",
        "role": "assistant"
      }
    }
  ],
  "created": 1723733419,
  "model": "llama3.1-8b",
  "system_fingerprint": "fp_70185065a4",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 10,
    "total_tokens": 22
  },
  "time_info": {
    "queue_time": 0.000073161,
    "prompt_time": 0.0010744798888888889,
    "completion_time": 0.005658071111111111,
    "total_time": 0.022224903106689453,
    "created": 1723733419
  }}"#;

        let now = Local::now().timestamp_millis();

        // Parse the string of data into serde_json::Value.
        let p: OpenAIGenerateResponse = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                println!("Error: {:?}", e);
                OpenAIGenerateResponse{
                    id: "".to_string(),
                    choices: vec![],
                    created: now,
                    model: "llama3.1-8b".to_string(),
                    system_fingerprint: "".to_string(),
                    object: "".to_string(),
                    usage: OpenAIGenerateResponseUsage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                    time_info: Some(OpenAIGenerateResponseTimeInfo {
                        queue_time: 0.0,
                        prompt_time: 0.0,
                        completion_time: 0.0,
                        total_time: 0.0,
                        created: now,
                    }),
                }
            },
        };

        println!("response: hi");
    }
}
