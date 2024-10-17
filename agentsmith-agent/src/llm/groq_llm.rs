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
use crate::llm::openai_llm::{OpenAIGenerateResponse, OpenAIRequest, OpenAIRequestMessage, UserContent};

#[derive(Clone, Debug)]
pub struct GroqLLM {
    api_key: String,
    base_url: String,
    model: String,
    client: Arc<Mutex<reqwest::Client>>,
}

impl GroqLLM {

    pub fn new(config: Config, model: String) -> Self {

        let groq_config = config.config.gateways.registry
            .get("groq_gateway")
            .unwrap();

        let api_key = groq_config.api_key.clone();
        let base_url = groq_config.baseurl.clone();
        let client = Arc::new(Mutex::new(reqwest::ClientBuilder::new()
            .connect_timeout(Duration::from_secs(60))
            .build()
            .unwrap()));

        Self { api_key, base_url, model, client }
    }


}

impl GenerateText for GroqLLM {

    async fn generate(&self, prompt: &Prompt) -> Result<LLMResult> {
        let url_str = format!("{}/{}", self.base_url.clone(), "openai/v1/chat/completions");
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
        //
        Ok(LLMResult::from_groq(&res))
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
                    time_info: None,
                }
            },
        };

        println!("response: {:?}", p);
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_tools_parse() {

        let data = r#"{
  "id": "chatcmpl-b9dd3ddc-1869-4cdd-adc3-ec1a7e6ab4e1",
  "object": "chat.completion",
  "created": 1729110815,
  "model": "llama3-8b-8192",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_yywq",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\":\"Boston, MA\",\"unit\":\"fahrenheit\"}"
            }
          }
        ]
      },
      "logprobs": null,
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "queue_time": 0.0021036009999999966,
    "prompt_tokens": 964,
    "prompt_time": 0.120271111,
    "completion_tokens": 77,
    "completion_time": 0.064166667,
    "total_tokens": 1041,
    "total_time": 0.184437778
  },
  "system_fingerprint": "fp_6a6771ae9c",
  "x_groq": {
    "id": "req_01jabgjc95e4evkhn6k654jk68"
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
                    time_info: None,
                }
            },
        };

        println!("response: {:?}", p);
    }
}
