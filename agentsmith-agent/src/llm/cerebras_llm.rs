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
use crate::llm::openai_llm::{OpenAIGenerateResponse, OpenAIRequest, OpenAIRequestMessage, UserContent};
use crate::llm::prompt::Prompt;

#[derive(Clone, Debug)]
pub struct CerebrasLLM {
    global_config: GatewayConfig,
    config: LLMConfiguration,
    client: Arc<Mutex<reqwest::Client>>,
}

impl CerebrasLLM {

    pub fn new(config: Config, llm_configuration: LLMConfiguration) -> Self {

        let cerebras_config = config.config.gateways.registry
            .get("cerebras_gateway")
            .unwrap()
            .clone();

        let client = Arc::new(Mutex::new(reqwest::ClientBuilder::new()
            .connect_timeout(Duration::from_secs(60))
            .build()
            .unwrap()));

        Self { global_config: cerebras_config, config: llm_configuration, client }
    }


}

impl GenerateText for CerebrasLLM {

    async fn generate(&self, prompt: &Prompt) -> Result<LLMResult> {

        let global_config = self.global_config.clone();
        let config = self.config.clone();

        let url_str = format!("{}{}", config.base_url.clone().unwrap_or(global_config.baseurl), "/v1/chat/completions");
        let api_key = config.credentials.api_key.clone();

        let request_obj = OpenAIRequest::from_prompt(&config, prompt);

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

        Ok(LLMResult::from_cerebras(&res))
    }
}


#[cfg(test)]
mod tests {
    use std::fs;
    use agentsmith_common::config::config::read_config;
    use crate::llm::openai_llm::{OpenAIGenerateResponseTimeInfo, OpenAIGenerateResponseUsage};
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
            },
        };

        println!("response: {:?}", p);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_tools_parse() {

        let data = r#"{
  "id": "chatcmpl-07468809-0b53-4e2e-9d71-95fcc03c167a",
  "choices": [
    {
      "finish_reason": "tool_calls",
      "index": 0,
      "message": {
        "tool_calls": [
          {
            "id": "94b7346ee",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\": \"Boston, MA\", \"unit\": \"fahrenheit\"}"
            }
          }
        ],
        "role": "assistant"
      }
    }
  ],
  "created": 1729111136,
  "model": "llama3.1-8b",
  "system_fingerprint": "fp_97b75e13af",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 255,
    "completion_tokens": 19,
    "total_tokens": 274
  },
  "time_info": {
    "queue_time": 1.7731e-05,
    "prompt_time": 0.015234503277777778,
    "completion_time": 0.009452399722222223,
    "total_time": 0.03803896903991699,
    "created": 1729111136
  }
}"#;

        let now = Local::now().timestamp_millis();

        // Parse the string of data into serde_json::Value.
        let p: OpenAIGenerateResponse = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                println!("Error: {:?}", e);
                OpenAIGenerateResponse{
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
            },
        };

        println!("response: {:?}", p);
    }
}
