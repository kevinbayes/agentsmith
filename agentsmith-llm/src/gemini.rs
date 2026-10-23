//! Gemini `generateContent` (`POST {base}/v1beta/models/{model}:generateContent`).
//!
//! The model id is part of the path, so it is taken from each request. The API
//! key is the `x-goog-api-key` header and is never placed in the query string.
//! Function calls have no id on the wire; the client assigns `call_1`, `call_2`,
//! and looks the name back up when it sends `functionResponse`.

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

pub struct Gemini {
    base: Url,
    api_key: String,
    client: reqwest::Client,
    timeout: Duration,
    retries: u32,
}

impl std::fmt::Debug for Gemini {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemini")
            .field("base", &self.base.as_str())
            .field("api_key", &"[redacted]")
            .field("timeout", &self.timeout)
            .field("retries", &self.retries)
            .finish_non_exhaustive()
    }
}

impl Gemini {
    pub fn new(
        base_url: impl AsRef<str>,
        api_key: impl Into<String>,
        timeout: Duration,
        retries: u32,
    ) -> Result<Self, Error> {
        let mut base = http_base(base_url.as_ref())?;
        base.set_query(None);
        base.set_fragment(None);
        let client = reqwest::Client::builder()
            .connect_timeout(timeout)
            .timeout(timeout)
            .build()
            .map_err(|err| llm_error(LlmErrorKind::Unavailable, format!("http client: {err}")))?;
        Ok(Self {
            base,
            api_key: api_key.into(),
            client,
            timeout,
            retries: retries.min(MAX_RETRIES),
        })
    }
}

#[async_trait]
impl Llm for Gemini {
    async fn complete(&self, request: CompletionRequest) -> Result<Completion, Error> {
        if request.cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let endpoint = generate_content_url(&self.base, &request.model)?;
        let body = build_body(&request)?;
        let text = post_json(
            &self.client,
            self.retries,
            self.timeout,
            &endpoint,
            None,
            &[("x-goog-api-key", self.api_key.as_str())],
            &body,
            &request.cancel,
        )
        .await?;
        parse_completion(&text)
    }
}

fn generate_content_url(base: &Url, model: &str) -> Result<Url, Error> {
    if model.is_empty() {
        return Err(llm_error(LlmErrorKind::BadRequest, "llm.model is required"));
    }
    if !model
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return Err(llm_error(
            LlmErrorKind::BadRequest,
            "llm.model contains unsupported characters",
        ));
    }
    let root = base.as_str().trim_end_matches('/');
    let joined = if root.ends_with("/v1beta") {
        format!("{root}/models/{model}:generateContent")
    } else {
        format!("{root}/v1beta/models/{model}:generateContent")
    };
    Url::parse(&joined).map_err(|err| Error::Config(format!("llm.base_url: {err}")))
}

fn build_body(request: &CompletionRequest) -> Result<Value, Error> {
    let mut generation = json!({
        "maxOutputTokens": request.max_output_tokens,
    });
    if let Some(temperature) = request.temperature {
        generation["temperature"] = finite_temperature(temperature)?;
    }
    let mut body = json!({
        "contents": contents_to_json(&request.messages),
        "generationConfig": generation,
    });
    let system = system_text(&request.messages);
    if !system.is_empty() {
        body["systemInstruction"] = json!({"parts": [{"text": system}]});
    }
    if !request.tools.is_empty() {
        body["tools"] = json!([{
            "functionDeclarations": request.tools.iter().map(tool_to_json).collect::<Vec<_>>(),
        }]);
        body["toolConfig"] = json!({
            "functionCallingConfig": tool_config(&request.tool_choice),
        });
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

fn contents_to_json(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    for message in messages {
        match message {
            Message::System { .. } => {}
            Message::User { content } => {
                push_parts(&mut out, "user", vec![json!({"text": content})]);
            }
            Message::Assistant {
                content,
                tool_calls,
            } => {
                let mut parts = Vec::new();
                if let Some(text) = content {
                    parts.push(json!({"text": text}));
                }
                for call in tool_calls {
                    parts.push(json!({
                        "functionCall": {
                            "name": call.name,
                            "args": object_arguments(call),
                        }
                    }));
                }
                if parts.is_empty() {
                    parts.push(json!({"text": ""}));
                }
                push_parts(&mut out, "model", parts);
            }
            Message::Tool {
                tool_call_id,
                content,
            } => {
                let name = tool_name(messages, tool_call_id);
                push_parts(
                    &mut out,
                    "user",
                    vec![json!({
                        "functionResponse": {
                            "name": name,
                            "response": {"output": content},
                        }
                    })],
                );
            }
        }
    }
    out
}

fn tool_name(messages: &[Message], id: &str) -> String {
    for message in messages.iter().rev() {
        if let Message::Assistant { tool_calls, .. } = message {
            if let Some(call) = tool_calls.iter().find(|call| call.id == id) {
                return call.name.clone();
            }
        }
    }
    id.to_string()
}

fn push_parts(out: &mut Vec<Value>, role: &str, parts: Vec<Value>) {
    if let Some(last) = out.last_mut() {
        if last.get("role").and_then(Value::as_str) == Some(role) {
            if let Some(existing) = last.get_mut("parts").and_then(Value::as_array_mut) {
                existing.extend(parts);
                return;
            }
        }
    }
    out.push(json!({"role": role, "parts": parts}));
}

fn tool_to_json(tool: &ToolDefinition) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    })
}

fn tool_config(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({"mode": "AUTO"}),
        ToolChoice::Function { name } => json!({
            "mode": "ANY",
            "allowedFunctionNames": [name],
        }),
    }
}

fn parse_completion(body: &str) -> Result<Completion, Error> {
    let value: Value = serde_json::from_str(body)
        .map_err(|_| llm_error(LlmErrorKind::Unavailable, "malformed gemini response"))?;
    if let Some(reason) = value
        .pointer("/promptFeedback/blockReason")
        .and_then(Value::as_str)
    {
        if value
            .get("candidates")
            .and_then(Value::as_array)
            .is_none_or(|items| items.is_empty())
        {
            let _ = reason;
            return Ok(Completion {
                message: Message::Assistant {
                    content: None,
                    tool_calls: Vec::new(),
                },
                stop: StopReason::ContentFilter,
            });
        }
    }
    let candidate = value.pointer("/candidates/0").ok_or_else(|| {
        llm_error(
            LlmErrorKind::Unavailable,
            "gemini response has no candidates",
        )
    })?;
    let parts = candidate
        .pointer("/content/parts")
        .and_then(Value::as_array);
    let mut text = String::new();
    let mut saw_text = false;
    let mut tool_calls = Vec::new();
    if let Some(parts) = parts {
        for part in parts {
            if let Some(chunk) = part.get("text").and_then(Value::as_str) {
                text.push_str(chunk);
                saw_text = true;
            }
            if let Some(call) = part.get("functionCall") {
                let index = tool_calls.len() + 1;
                if let Some(parsed) = parse_function_call(call, index) {
                    tool_calls.push(parsed);
                }
            }
        }
    }
    let finish = candidate.get("finishReason").and_then(Value::as_str);
    let has_tool_calls = !tool_calls.is_empty();
    Ok(Completion {
        message: Message::Assistant {
            content: saw_text.then_some(text),
            tool_calls,
        },
        stop: map_stop(finish, has_tool_calls),
    })
}

fn parse_function_call(call: &Value, index: usize) -> Option<ToolCall> {
    if !call.is_object() {
        return None;
    }
    let name = call
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let id = call
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("call_{index}"));
    let (arguments_raw, arguments) = match call.get("args") {
        None => ("{}".to_string(), Some(json!({}))),
        Some(args) => split_arguments(Some(args)),
    };
    Some(ToolCall {
        id,
        name,
        arguments_raw,
        arguments,
    })
}

fn map_stop(finish_reason: Option<&str>, has_tool_calls: bool) -> StopReason {
    match finish_reason {
        Some("MAX_TOKENS") => StopReason::Length,
        Some("SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII") => StopReason::ContentFilter,
        _ if has_tool_calls => StopReason::ToolUse,
        _ => StopReason::EndTurn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn puts_the_model_in_the_generate_content_path() {
        let base = http_base("https://generativelanguage.googleapis.com").unwrap();
        let url = generate_content_url(&base, "gemini-3.5-flash").unwrap();
        assert_eq!(
            url.as_str(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"
        );
        let prefixed = http_base("http://127.0.0.1:9/v1beta/").unwrap();
        let url = generate_content_url(&prefixed, "gemini-test").unwrap();
        assert_eq!(
            url.as_str(),
            "http://127.0.0.1:9/v1beta/models/gemini-test:generateContent"
        );
    }

    #[test]
    fn rejects_a_model_that_would_change_the_path() {
        let base = http_base("https://generativelanguage.googleapis.com").unwrap();
        let err = generate_content_url(&base, "gemini/flash").unwrap_err();
        assert!(matches!(err, Error::Llm { .. }), "{err}");
        let err = generate_content_url(&base, "").unwrap_err();
        assert!(matches!(err, Error::Llm { .. }), "{err}");
    }
}
