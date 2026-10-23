//! Anthropic Messages API (`POST {base}/v1/messages`).
//!
//! System text is the top-level `system` field. Tool results are `tool_result`
//! blocks on a user message. `tool_choice` `auto` or `{type: tool, name}` matches
//! the harness choice. The API key is the `x-api-key` header.

use std::time::Duration;

use agentsmith_core::{Error, LlmErrorKind};
use async_trait::async_trait;
use serde_json::{json, Value};
use url::Url;

use crate::arguments::{object_arguments, split_arguments};
use crate::http::{http_base, llm_error, post_json, MAX_RETRIES};
use crate::types::{
    Completion, CompletionRequest, Llm, Message, StopReason, ToolCall, ToolChoice, ToolDefinition,
};

const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct Anthropic {
    endpoint: Url,
    api_key: String,
    client: reqwest::Client,
    timeout: Duration,
    retries: u32,
}

impl std::fmt::Debug for Anthropic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Anthropic")
            .field("endpoint", &self.endpoint.as_str())
            .field("api_key", &"[redacted]")
            .field("timeout", &self.timeout)
            .field("retries", &self.retries)
            .finish_non_exhaustive()
    }
}

impl Anthropic {
    pub fn new(
        base_url: impl AsRef<str>,
        api_key: impl Into<String>,
        timeout: Duration,
        retries: u32,
    ) -> Result<Self, Error> {
        let client = reqwest::Client::builder()
            .connect_timeout(timeout)
            .timeout(timeout)
            .build()
            .map_err(|err| llm_error(LlmErrorKind::Unavailable, format!("http client: {err}")))?;
        Ok(Self {
            endpoint: messages_url(base_url.as_ref())?,
            api_key: api_key.into(),
            client,
            timeout,
            retries: retries.min(MAX_RETRIES),
        })
    }
}

#[async_trait]
impl Llm for Anthropic {
    async fn complete(&self, request: CompletionRequest) -> Result<Completion, Error> {
        if request.cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let body = build_body(&request)?;
        let text = post_json(
            &self.client,
            self.retries,
            self.timeout,
            &self.endpoint,
            None,
            &[
                ("x-api-key", self.api_key.as_str()),
                ("anthropic-version", ANTHROPIC_VERSION),
            ],
            &body,
            &request.cancel,
        )
        .await?;
        parse_completion(&text)
    }
}

fn messages_url(base_url: &str) -> Result<Url, Error> {
    let mut url = http_base(base_url)?;
    url.set_query(None);
    url.set_fragment(None);
    let text = url.as_str().trim_end_matches('/').to_string();
    let joined = if text.ends_with("/v1/messages") {
        text
    } else if text.ends_with("/v1") {
        format!("{text}/messages")
    } else {
        format!("{text}/v1/messages")
    };
    Url::parse(&joined).map_err(|err| Error::Config(format!("llm.base_url: {err}")))
}

fn build_body(request: &CompletionRequest) -> Result<Value, Error> {
    let mut body = json!({
        "model": request.model,
        "max_tokens": request.max_output_tokens,
        "messages": messages_to_json(&request.messages),
    });
    let system = system_text(&request.messages);
    if !system.is_empty() {
        body["system"] = Value::String(system);
    }
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(request.tools.iter().map(tool_to_json).collect());
        body["tool_choice"] = tool_choice_to_json(&request.tool_choice);
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = finite_temperature(temperature)?;
    }
    Ok(body)
}

fn finite_temperature(temperature: f32) -> Result<Value, Error> {
    if !temperature.is_finite() {
        return Err(llm_error(
            LlmErrorKind::BadRequest,
            "temperature must be finite",
        ));
    }
    let number = serde_json::Number::from_f64(f64::from(temperature))
        .ok_or_else(|| llm_error(LlmErrorKind::BadRequest, "temperature must be finite"))?;
    Ok(Value::Number(number))
}

fn system_text(messages: &[Message]) -> String {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::System { content } => Some(content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn messages_to_json(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    for message in messages {
        match message {
            Message::System { .. } => {}
            Message::User { content } => push_blocks(
                &mut out,
                "user",
                vec![json!({"type": "text", "text": content})],
            ),
            Message::Assistant {
                content,
                tool_calls,
            } => {
                let mut blocks = Vec::new();
                if let Some(text) = content {
                    blocks.push(json!({"type": "text", "text": text}));
                }
                for call in tool_calls {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": object_arguments(call),
                    }));
                }
                if blocks.is_empty() {
                    blocks.push(json!({"type": "text", "text": ""}));
                }
                push_blocks(&mut out, "assistant", blocks);
            }
            Message::Tool {
                tool_call_id,
                content,
            } => push_blocks(
                &mut out,
                "user",
                vec![json!({
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                })],
            ),
        }
    }
    out
}

fn push_blocks(out: &mut Vec<Value>, role: &str, blocks: Vec<Value>) {
    if let Some(last) = out.last_mut() {
        if last.get("role").and_then(Value::as_str) == Some(role) {
            if let Some(content) = last.get_mut("content").and_then(Value::as_array_mut) {
                content.extend(blocks);
                return;
            }
        }
    }
    out.push(json!({"role": role, "content": blocks}));
}

fn tool_to_json(tool: &ToolDefinition) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    })
}

fn tool_choice_to_json(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({"type": "auto"}),
        ToolChoice::Function { name } => json!({"type": "tool", "name": name}),
    }
}

fn parse_completion(body: &str) -> Result<Completion, Error> {
    let value: Value = serde_json::from_str(body)
        .map_err(|_| llm_error(LlmErrorKind::Unavailable, "malformed anthropic message"))?;
    let blocks = value
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            llm_error(
                LlmErrorKind::Unavailable,
                "anthropic message is missing content",
            )
        })?;
    let mut text = String::new();
    let mut saw_text = false;
    let mut tool_calls = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(part) = block.get("text").and_then(Value::as_str) {
                    text.push_str(part);
                    saw_text = true;
                }
            }
            Some("tool_use") => {
                if let Some(call) = parse_tool_use(block) {
                    tool_calls.push(call);
                }
            }
            _ => {}
        }
    }
    let stop_reason = value.get("stop_reason").and_then(Value::as_str);
    let has_tool_calls = !tool_calls.is_empty();
    Ok(Completion {
        message: Message::Assistant {
            content: saw_text.then_some(text),
            tool_calls,
        },
        stop: map_stop(stop_reason, has_tool_calls),
    })
}

fn parse_tool_use(block: &Value) -> Option<ToolCall> {
    if !block.is_object() {
        return None;
    }
    let id = block
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let name = block
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let (arguments_raw, arguments) = split_arguments(block.get("input"));
    Some(ToolCall {
        id,
        name,
        arguments_raw,
        arguments,
    })
}

/// `max_tokens` and `refusal` win even when tool calls are present.
fn map_stop(stop_reason: Option<&str>, has_tool_calls: bool) -> StopReason {
    match stop_reason {
        Some("max_tokens") => StopReason::Length,
        Some("refusal") => StopReason::ContentFilter,
        _ if has_tool_calls => StopReason::ToolUse,
        _ => StopReason::EndTurn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_messages_without_a_second_v1() {
        let cases = [
            (
                "https://api.anthropic.com",
                "https://api.anthropic.com/v1/messages",
            ),
            (
                "https://api.anthropic.com/",
                "https://api.anthropic.com/v1/messages",
            ),
            (
                "https://api.anthropic.com/v1",
                "https://api.anthropic.com/v1/messages",
            ),
            (
                "https://api.anthropic.com/v1/",
                "https://api.anthropic.com/v1/messages",
            ),
            (
                "https://api.anthropic.com/v1/messages",
                "https://api.anthropic.com/v1/messages",
            ),
            (
                "http://127.0.0.1:9/proxy",
                "http://127.0.0.1:9/proxy/v1/messages",
            ),
        ];
        for (base, expected) in cases {
            assert_eq!(messages_url(base).unwrap().as_str(), expected, "{base}");
        }
    }

    #[test]
    fn strips_a_query_so_the_key_is_not_in_the_url() {
        let url = messages_url("https://api.anthropic.com/v1?key=secret").unwrap();
        assert_eq!(url.as_str(), "https://api.anthropic.com/v1/messages");
        assert!(url.query().is_none());
    }
}
