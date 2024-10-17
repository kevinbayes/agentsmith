use crate::llm::llm::{GenerateText, LLMResult, Prompt};
use agentsmith_common::config::config::Config;
use agentsmith_common::error::error::{Error, Result};
use chrono::Local;
use futures_util::TryFutureExt;
use reqwest::{Proxy, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::Duration;
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIRequestMessage {
    System {
        role: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        role: String,
        content: Vec<UserContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        role: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<Vec<AssistantContent>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<AssistantToolCall>,
    },
    Tool { role: String, content: Vec<String>, name: String },
}


#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text {
        #[serde(rename = "type")]
        type_: String,
        text: String,
    },
    Image {
        #[serde(rename = "type")]
        type_: String,
        image_url: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AssistantContent {
    Text {
        #[serde(rename = "type")]
        type_: String,
        text: String,
    },
    Image {
        #[serde(rename = "type")]
        type_: String,
        refusal: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssistantToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub function: Value,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OpenAIGenerateResponse {
    pub id: Option<String>,
    pub choices: Vec<OpenAIGenerateResponseChoice>,
    pub created: i64,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: OpenAIGenerateResponseUsage,
    pub time_info: Option<OpenAIGenerateResponseTimeInfo>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OpenAIGenerateResponseChoice {
    pub finish_reason: String,
    pub index: u32,
    pub message: OpenAIGenerateResponseMessage,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OpenAIGenerateResponseMessage {
    pub content: Option<String>,
    pub role: String,
    pub tool_calls: Option<Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OpenAIGenerateResponseUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

    async fn generate(&self, prompt: &Prompt) -> Result<LLMResult> {

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

        let request = client.post(&url_str)
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
    use super::*;


    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_parse() {
        let data = r#"{
  "id": "chatcmpl-AJ56iSaW6FzvMSON6MHVEUkUIMKOV",
  "object": "chat.completion",
  "created": 1729111228,
  "model": "gpt-4o-mini-2024-07-18",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Whispers of circuits,  \nDreams in lines of code awaken,  \nMind of silicon.",
        "refusal": null
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 19,
    "total_tokens": 32,
    "prompt_tokens_details": {
      "cached_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0
    }
  },
  "system_fingerprint": "fp_e2bde53e6e"
}"#;

        let now = Local::now().timestamp_millis();

        // Parse the string of data into serde_json::Value.
        let p: OpenAIGenerateResponse = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                println!("Error: {:?}", e);
                OpenAIGenerateResponse {
                    id: None,
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
            }
        };

        println!("response: {:?}", p);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_tools_parse() {
        let data = r#"{
  "id": "chatcmpl-AJ59DHZQNyOrnwnaoMCmsTDNv0Zau",
  "object": "chat.completion",
  "created": 1729111383,
  "model": "gpt-4o-mini-2024-07-18",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_GqTMRz3Db3NfRS1y3m9RdA1X",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\":\"Boston, MA\"}"
            }
          }
        ],
        "refusal": null
      },
      "logprobs": null,
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 80,
    "completion_tokens": 17,
    "total_tokens": 97,
    "prompt_tokens_details": {
      "cached_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0
    }
  },
  "system_fingerprint": "fp_e2bde53e6e"
}"#;

        let now = Local::now().timestamp_millis();

        // Parse the string of data into serde_json::Value.
        let p: OpenAIGenerateResponse = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                println!("Error: {:?}", e);
                OpenAIGenerateResponse {
                    id: None,
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
            }
        };

        println!("response: {:?}", p);
    }
}
