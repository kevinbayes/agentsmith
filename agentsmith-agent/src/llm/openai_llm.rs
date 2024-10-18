use crate::llm::llm::{GenerateText, LLMConfiguration, LLMResult};
use agentsmith_common::config::config::{Config, GatewayConfig};
use agentsmith_common::error::error::{Error, Result};
use chrono::Local;
use futures_util::TryFutureExt;
use reqwest::{Proxy, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use r2d2_redis::redis::Commands;
use tracing::info;
use tracing_subscriber;
use crate::llm::prompt::{Prompt, PromptMessage, UserContent as PromptUserContent, AssistantContent as PromptAssistantContent, AssistantToolCall as PromptAssistantToolCall, Tool as PromptTool, ToolChoice as PromptToolChoice};

#[derive(Clone, Debug)]
pub struct OpenAILLM {
    global_config: GatewayConfig,
    config: LLMConfiguration,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

impl OpenAIRequest {

    pub(crate) fn from_prompt(config: &LLMConfiguration, prompt: &Prompt) -> Self {

        match prompt {
            Prompt::Simple { system, user, tools, tool_choice } => {

                let system = system.clone();
                let user = user.clone();

                OpenAIRequest {
                    model: config.model.clone(),
                    stream: config.stream.clone().unwrap_or(false),
                    messages: vec![OpenAIRequestMessage::System {
                        role: String::from("system"),
                        content: system,
                        name: None,
                    }, OpenAIRequestMessage::User {
                        role: String::from("user"),
                        content: vec![UserContent::Text { type_: "text".to_string(), text: user }],
                        name: None,
                    }],
                    temperature: config.temperature.clone().unwrap_or(1.0),
                    max_tokens: config.max_tokens.clone().unwrap_or(500),
                    seed: config.seed.clone().unwrap_or(0),
                    top_p: config.top_p.clone().unwrap_or(1),
                    tool_choice: None,
                    tools: None,
                    parallel_tool_calls: None,
                }
            }
            Prompt::Messages { system, messages, tools, tool_choice } => {

                let system = system.clone();
                let messages = messages.clone();

                let mut request_messages = vec![
                    OpenAIRequestMessage::System {
                        role: String::from("system"),
                        content: system,
                        name: None,
                    }
                ];

                let other_messages: Vec<OpenAIRequestMessage> = messages.iter().map(OpenAIRequestMessage::from_prompt_message).collect();
                request_messages.extend(other_messages);

                let converted_tools = &if tools.is_none() {
                    None
                } else {
                    Some(tools.clone().unwrap().iter().map(|item| Tool::from_prompt(item)).collect())
                };

                let mut parallel_tool_calls: Option<bool> = None;

                let is_empty_tools = converted_tools.clone().unwrap_or(vec![]).is_empty();

                let tool_choice = if is_empty_tools {
                    None
                } else {
                    match tool_choice {
                        Some(choice) => {
                            match choice.clone() {
                                PromptToolChoice::Any { disable_parallel_tool_use, .. } => {
                                    parallel_tool_calls = disable_parallel_tool_use;
                                    None
                                },
                                PromptToolChoice::Auto { disable_parallel_tool_use, .. } => {
                                    parallel_tool_calls = disable_parallel_tool_use;
                                    None
                                },
                                PromptToolChoice::Required { type_, disable_parallel_tool_use } => {
                                    parallel_tool_calls = disable_parallel_tool_use;
                                    Some(ToolChoice::String("required".to_string()))
                                },
                                PromptToolChoice::Tool { type_, name, disable_parallel_tool_use } => {
                                    parallel_tool_calls = disable_parallel_tool_use;
                                    Some(ToolChoice::Function {
                                        type_: "function".to_string(),
                                        name: ToolChoiceFunctionName { name },
                                    })
                                }
                            }
                        },
                        None => None
                    }
                };


                OpenAIRequest {
                    model: config.model.clone(),
                    stream: config.stream.clone().unwrap_or(false),
                    messages: request_messages,
                    temperature: config.temperature.clone().unwrap_or(1.0),
                    max_tokens: config.max_tokens.clone().unwrap_or(500),
                    seed: config.seed.clone().unwrap_or(0),
                    top_p: config.top_p.clone().unwrap_or(1),
                    tool_choice: tool_choice,
                    tools: converted_tools.clone(),
                    parallel_tool_calls: parallel_tool_calls,
                }
            }
        }
    }
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

impl OpenAIRequestMessage {

    pub fn from_prompt_message(prompt: &PromptMessage) -> Self {
        let prompt = prompt.clone();
        match prompt {
            PromptMessage::System { role, content, name } => {
                OpenAIRequestMessage::System { role, content, name }
            }
            PromptMessage::User { role, content, name } => {
                let content = content.iter().map(UserContent::map_user_content).collect();
                OpenAIRequestMessage::User { role, content, name }
            }
            PromptMessage::Assistant { role, content, refusal, name, tool_calls } => {
                let content_content = if content.is_none() {
                    None
                } else {
                    let content = content.unwrap().iter().map(AssistantContent::from_prompt_assistant_content).collect();
                    Some(content)
                };

                let tool_calls = if tool_calls.is_none() {
                    None
                } else {
                    Some(AssistantToolCall::from_prompt_assistant_tool_calls(tool_calls.unwrap()))
                };

                OpenAIRequestMessage::Assistant { role, content: content_content, refusal, name, tool_calls }
            }
            PromptMessage::Tool { role, content, name } => {
                OpenAIRequestMessage::Tool { role, content, name }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContentImageUrl {
    pub url: String,
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
        image_url: ContentImageUrl,
    },
}

impl UserContent {

    fn map_user_content(user_content: &PromptUserContent) -> Self {

        match user_content {
            PromptUserContent::Text { type_, text } => {
                Self::Text {
                    type_: type_.clone(),
                    text: text.clone()
                }
            }
            PromptUserContent::Image { type_, content_type, image_url } => {
                Self::Image {
                    type_: type_.clone(),
                    image_url: ContentImageUrl { url: image_url.url.clone() },
                }
            }
        }
    }
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
        image_url: ContentImageUrl,
    },
}

impl AssistantContent {
    pub fn from_prompt_assistant_content(content: &PromptAssistantContent) -> Self {

        match content {
            PromptAssistantContent::Text { type_, text } => {
                AssistantContent::Text {
                    type_: type_.clone(),
                    text: text.clone()
                }
            }
            PromptAssistantContent::Image { type_, content_type, image_url } => {
                AssistantContent::Image {
                    type_: type_.clone(),
                    image_url: ContentImageUrl { url: image_url.url.clone() },
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssistantToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub function: Value,
}

impl AssistantToolCall {

    pub fn from_prompt_assistant_tool_calls(tool_call: PromptAssistantToolCall) -> Self {

        Self {
            function: tool_call.function.clone(),
            id: tool_call.id,
            type_: tool_call.type_,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    String(String),
    Function {
        #[serde(rename = "type")]
        type_: String,
        name: ToolChoiceFunctionName,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolChoiceFunctionName {
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    pub type_: String,
    pub function: ToolFunction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl Tool {
    fn from_prompt(item: &PromptTool) -> Self {
        Self {
            type_: "function".to_string(),
            function: ToolFunction {
                name: item.name.clone(),
                description: item.description.clone(),
                parameters: item.input_schema.clone(),
            }
        }
    }
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
    pub fn new(config: Config, llm_config: LLMConfiguration) -> Self {

        let openai_config = config.config.gateways.registry
            .get("openai_gateway")
            .unwrap()
            .clone();

        let client = Arc::new(Mutex::new(reqwest::ClientBuilder::new()
            .connect_timeout(Duration::from_secs(60))
            .build()
            .unwrap()));

        Self { global_config: openai_config, config: llm_config, client }
    }
}

impl GenerateText for OpenAILLM {

    async fn generate(&self, prompt: &Prompt) -> Result<LLMResult> {

        let global_config = self.global_config.clone();
        let config = self.config.clone();

        let url_str = format!("{}{}", config.base_url.clone().unwrap_or(global_config.baseurl), "/v1/chat/completions");
        let api_key = config.credentials.api_key.clone();

        let request_obj = OpenAIRequest::from_prompt(&config, prompt);

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
