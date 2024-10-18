use std::sync::Arc;
use std::time::Duration;
use futures_util::TryFutureExt;
use reqwest::{Proxy, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use agentsmith_common::config::config::{Config, GatewayConfig};
use crate::llm::llm::{GenerateText, LLMConfiguration, LLMResult};

use agentsmith_common::error::error::{Error, Result};
use crate::llm::openai_llm::{OpenAIRequest, OpenAIRequestMessage, UserContent};
use crate::llm::prompt::Prompt;

#[derive(Clone, Debug)]
pub struct GeminiLLM {
    global_config: GatewayConfig,
    config: LLMConfiguration,
    client: Arc<reqwest::Client>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GeminiGenerateRequest {
    #[serde(rename = "contents")]
    pub contents: Vec<GeminiGenerateRequestContent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GeminiGenerateRequestContent {
    #[serde(rename = "parts")]
    pub parts: Vec<GeminiGenerateRequestContentPart>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GeminiGenerateRequestContentPart {
    #[serde(rename = "text")]
    pub text: String,
}


#[derive(Clone, Debug, Serialize, Deserialize)]
struct GeminiGenerateResponse {
    pub candidates: Vec<GeminiGenerateResponseCandidate>,
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: UsageMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UsageMetadata {
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: i64,
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: i64,
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GeminiGenerateResponseCandidate {
    pub content: GeminiGenerateRequestContent,
    #[serde(rename = "finishReason")]
    pub finish_reason: String,
    #[serde(rename = "index")]
    pub index: u32,
}


#[derive(Clone, Debug, Serialize, Deserialize)]
struct GeminiGenerateResponseCandidateContent {
    pub role: String,
    #[serde(rename = "parts")]
    pub parts: Vec<GeminiGenerateResponseCandidateContentPart>,
}


#[derive(Clone, Debug, Serialize, Deserialize)]
struct GeminiGenerateResponseCandidateContentPart {
    pub text: String
}

impl GeminiLLM {

    pub fn new(config: Config, llm_configuration: LLMConfiguration) -> Self {

        let gemini_config = config.config.gateways.registry
            .get("gemini_gateway")
            .unwrap()
            .clone();

        let client = Arc::new(reqwest::ClientBuilder::new()
            .connect_timeout(Duration::from_secs(60))
            .build()
            .unwrap());

        Self { global_config: gemini_config, config: llm_configuration, client }
    }
}

impl GenerateText for GeminiLLM {

    async fn generate(&self, prompt: &Prompt) -> Result<LLMResult> {

        let global_config = self.global_config.clone();
        let config = self.config.clone();

        // let url_str = format!("{}{}?key={}", &self.base_url, "/v1beta/models/gemini-pro:generateText", &self.api_key.clone());
        let url_str = format!("{}{}", config.base_url.unwrap_or(global_config.baseurl), "/v1beta/models/gemini-pro:generateContent");
        let api_key = config.credentials.api_key.clone();


        let prompt = match prompt {
            Prompt::Simple { system, user, tools, tool_choice } => {

                let system = system.clone();
                let user = user.clone();

                format!("\n{}\n{}\n", system, user)
            }
            Prompt::Messages { system, messages, tools, tool_choice } => {

                format!("")
            }
        };

        if prompt.len() == 0 {
            Err(Error::LLMError { id: 0, code: 0 })
        } else {

            let request = GeminiGenerateRequest {
                contents: vec![GeminiGenerateRequestContent {
                    parts: vec![GeminiGenerateRequestContentPart {
                        text: prompt,
                    }],
                }],
            };


            let res = self.client.post(url_str)
                .query(&[("key", api_key)])
                .json(&request)
                .send()
                .map_err(|e| {
                    println!("Error: {:?}", e);
                    Error::AgentError { id: 0, code: 1 }
                })
                .await?
                .json::<GeminiGenerateResponse>()
                .map_err(|e| {
                    println!("Error: {:?}", e);
                    Error::AgentError { id: 0, code: 2 }
                })
                .await?;

            let response = &res.candidates[0].content.parts[0].text;

            Ok(LLMResult::new(response.clone()))
        }
    }
}


#[cfg(test)]
mod tests {
    use std::fs;
    use agentsmith_common::config::config::read_config;
    use super::*;



    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_parse() {

        let data = r#"{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "test\ntest"
          }
        ],
        "role": "model"
      },
      "finishReason": "STOP",
      "index": 0,
      "safetyRatings": [
        {
          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
          "probability": "NEGLIGIBLE"
        },
        {
          "category": "HARM_CATEGORY_HATE_SPEECH",
          "probability": "NEGLIGIBLE"
        },
        {
          "category": "HARM_CATEGORY_HARASSMENT",
          "probability": "NEGLIGIBLE"
        },
        {
          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
          "probability": "NEGLIGIBLE"
        }
      ]
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 5,
    "candidatesTokenCount": 3,
    "totalTokenCount": 8
  }
}"#;

        // Parse the string of data into serde_json::Value.
        let p: GeminiGenerateResponse = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                println!("Error: {:?}", e);
                GeminiGenerateResponse{candidates: vec![], usage_metadata: UsageMetadata{prompt_token_count: 0, candidates_token_count: 0, total_token_count: 0}}
            },
        };

        println!("response: hi");
    }
}
