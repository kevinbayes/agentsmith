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
pub struct CerebrasLLM {
    api_key: String,
    base_url: String,
    model: String,
    client: Arc<Mutex<reqwest::Client>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CerebrasRequest {
    #[serde(rename = "model")]
    pub model: String,
    #[serde(rename = "stream")]
    pub stream: bool,
    #[serde(rename = "messages")]
    pub messages: Vec<CerebrasRequestMessage>,
    #[serde(rename = "temperature")]
    pub temperature: f32,
    #[serde(rename = "max_tokens")]
    pub max_tokens: i32,
    #[serde(rename = "seed")]
    pub seed: i32,
    #[serde(rename = "top_p")]
    pub top_p: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CerebrasRequestMessage {
    #[serde(rename = "content")]
    pub content: String,
    #[serde(rename = "role")]
    pub role: String,
}


#[derive(Debug, Deserialize, Serialize)]
pub struct CerebrasGenerateResponse {
    pub id: String,
    pub choices: Vec<CerebrasGenerateResponseChoice>,
    pub created: i64,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: CerebrasGenerateResponseUsage,
    pub time_info: CerebrasGenerateResponseTimeInfo,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CerebrasGenerateResponseChoice {
    pub finish_reason: String,
    pub index: u32,
    pub message: CerebrasGenerateResponseMessage,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CerebrasGenerateResponseMessage {
    pub content: String,
    pub role: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CerebrasGenerateResponseUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CerebrasGenerateResponseTimeInfo {
    pub queue_time: f64,
    pub prompt_time: f64,
    pub completion_time: f64,
    pub total_time: f64,
    pub created: i64,
}


impl CerebrasLLM {

    pub fn new(config: Config, model: String) -> Self {

        let cerebras_config = config.config.gateways.registry
            .get("cerebras_gateway")
            .unwrap();

        let api_key = cerebras_config.api_key.clone();
        let base_url = cerebras_config.baseurl.clone();
        let client = Arc::new(Mutex::new(reqwest::ClientBuilder::new()
            .connect_timeout(Duration::from_secs(60))
            .build()
            .unwrap()));

        Self { api_key, base_url, model, client }
    }


}

impl GenerateText for CerebrasLLM {

    async fn generate_text(&self, prompt: &Prompt) -> Result<LLMResult> {
        let url_str = format!("{}{}", self.base_url.clone(), "/v1/chat/completions");
        let api_key = self.api_key.clone();
        let model = self.model.clone();

        let request_obj = CerebrasRequest {
            model,
            stream: false,
            messages: vec![CerebrasRequestMessage {
                role: String::from("system"),
                content: prompt.clone().system,
            }, CerebrasRequestMessage {
                role: String::from("user"),
                content: prompt.clone().user,
            }],
            temperature: 1.0,
            max_tokens: 500,
            seed: 0,
            top_p: 1,
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
            .json::<CerebrasGenerateResponse>()
            .map_err(|e| {
                println!("Error: {:?}", e);
                Error::AgentError { id: 0, code: 2 }
            })
            .await?;

        let response = &res.choices[0].message.content;
        //
        Ok(LLMResult::new(response.clone()))
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
        let p: CerebrasGenerateResponse = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                println!("Error: {:?}", e);
                CerebrasGenerateResponse{
                    id: "".to_string(),
                    choices: vec![],
                    created: now,
                    model: "llama3.1-8b".to_string(),
                    system_fingerprint: "".to_string(),
                    object: "".to_string(),
                    usage: CerebrasGenerateResponseUsage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                    time_info: CerebrasGenerateResponseTimeInfo {
                        queue_time: 0.0,
                        prompt_time: 0.0,
                        completion_time: 0.0,
                        total_time: 0.0,
                        created: now,
                    },
                }
            },
        };

        println!("response: hi");
    }
}
