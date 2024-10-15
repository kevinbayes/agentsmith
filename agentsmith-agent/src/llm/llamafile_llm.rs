use std::sync::Arc;
use std::time::Duration;
use futures_util::TryFutureExt;
use reqwest::{Proxy, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use crate::config::config::Config;
use crate::llm::llm::{GenerateText, LLMResult, Prompt};
use crate::error::{Error, Result};

#[derive(Clone, Debug)]
pub struct LlamafileLLM {
    api_key: String,
    base_url: String,
    client: Arc<reqwest::Client>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LlamafileLLMRequest {
    #[serde(rename = "model")]
    pub model: String,
    #[serde(rename = "messages")]
    pub messages: Vec<LlamafileMessage>,
    #[serde(rename = "stream")]
    pub stream: bool,
    #[serde(rename = "max_tokens")]
    pub max_tokens: i64,
    #[serde(rename = "temperature")]
    pub temperature: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LlamafileMessage {
    #[serde(rename = "role")]
    pub role: String,
    #[serde(rename = "content")]
    pub content: String,
}


#[derive(Clone, Debug, Serialize, Deserialize)]
struct LlamafileResponse {
    #[serde(rename = "id")]
    pub id: String,
    #[serde(rename = "choices")]
    pub choices: String,
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

    pub fn new(config: Config) -> Self {

        let gemini_config = config.config.gateways.registry
            .get("gemini_gateway")
            .unwrap();

        let api_key = gemini_config.api_key.clone();
        let base_url = gemini_config.baseurl.clone();
        let client = Arc::new(reqwest::ClientBuilder::new()
            .connect_timeout(Duration::from_secs(60))
            .build()
            .unwrap());

        Self { base_url, api_key, client }
    }


}

impl GenerateText for GeminiLLM {

    async fn generate_text(&self, prompt: Prompt) -> Result<LLMResult> {
        let url_str = format!("{}{}?key={}", &self.base_url, "/v1beta/models/gemini-pro:generateText", &self.api_key.clone());
        let url_str = format!("{}{}", &self.base_url, "/v1beta/models/gemini-pro:generateContent");

        let prompt_str = format!("\n{}\n{}\n", prompt.system, prompt.user);

        let request = GeminiGenerateRequest {
            contents: vec![GeminiGenerateRequestContent {
                parts: vec![GeminiGenerateRequestContentPart {
                    text: prompt_str,
                }],
            }],
        };


        let res = self.client.post(url_str)
            .query(&[("key", &self.api_key.clone())])
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
        //
        Ok(LLMResult::new(response.clone()))
    }
}


#[cfg(test)]
mod tests {
    use std::fs;
    use crate::config::config::read_yaml_config;
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
