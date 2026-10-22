use std::time::Duration;

use agentsmith_core::{Error, LlmErrorKind};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::types::{
    Completion, CompletionRequest, Llm, Message, StopReason, ToolCall, ToolChoice, ToolDefinition,
};

/// Extra attempts after the first. Matches the config ceiling in `agentsmith-core`.
const MAX_RETRIES: u32 = 4;
const BACKOFF_FIRST_MS: u64 = 500;
const BACKOFF_LATER_MS: u64 = 1500;

/// OpenAI chat-completions client (`POST {base_url}/chat/completions`).
///
/// The API key is sent only as `Authorization`. Neither it nor the request body
/// is logged or copied into errors.
pub struct OpenAiCompatible {
    endpoint: Url,
    api_key: String,
    client: reqwest::Client,
    timeout: Duration,
    retries: u32,
}

impl std::fmt::Debug for OpenAiCompatible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatible")
            .field("endpoint", &self.endpoint.as_str())
            .field("api_key", &"[redacted]")
            .field("timeout", &self.timeout)
            .field("retries", &self.retries)
            .finish_non_exhaustive()
    }
}

impl OpenAiCompatible {
    /// `retries` is the number of extra attempts after the first and is capped at 4.
    ///
    /// 429, 5xx, transport failures, and timeouts wait 500ms, then 1500ms, with
    /// ±20% jitter. 401 and 403 (`Auth`) and other 4xx (`BadRequest`, except 429)
    /// are not retried.
    pub fn new(
        base_url: impl AsRef<str>,
        api_key: impl Into<String>,
        timeout: Duration,
        retries: u32,
    ) -> Result<Self, Error> {
        let endpoint = chat_completions_url(base_url.as_ref())?;
        let client = reqwest::Client::builder()
            .connect_timeout(timeout)
            .timeout(timeout)
            .build()
            .map_err(|err| llm_error(LlmErrorKind::Unavailable, format!("http client: {err}")))?;
        Ok(Self {
            endpoint,
            api_key: api_key.into(),
            client,
            timeout,
            retries: retries.min(MAX_RETRIES),
        })
    }

    async fn once(
        &self,
        body: &Value,
        cancel: &CancellationToken,
    ) -> Result<Completion, OnceError> {
        if cancel.is_cancelled() {
            return Err(OnceError::Cancelled);
        }

        // `body` and `api_key` stay off the error path and out of any log.
        let pending = self
            .client
            .post(self.endpoint.clone())
            .bearer_auth(&self.api_key)
            .timeout(self.timeout)
            .json(body)
            .send();

        let response = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(OnceError::Cancelled),
            result = pending => result,
        };

        let response = match response {
            Ok(response) => response,
            Err(err) => return Err(transport_error(&err)),
        };

        let status = response.status();
        let body_result = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(OnceError::Cancelled),
            result = response.text() => result,
        };

        if !status.is_success() {
            // Drop the server body. A 400 can echo the prompt, and the body can
            // echo the key. Callers log `Error`.
            drop(body_result);
            let message = format!("HTTP {status}");
            let (kind, retry) = classify_status(status);
            let err = llm_error(kind, message);
            return Err(if retry {
                OnceError::Retry(err)
            } else {
                OnceError::Done(err)
            });
        }

        let text = match body_result {
            Ok(text) => text,
            Err(err) => return Err(transport_error(&err)),
        };
        match parse_completion(&text) {
            Ok(completion) => Ok(completion),
            Err(err) => Err(OnceError::Done(err)),
        }
    }
}

#[async_trait]
impl Llm for OpenAiCompatible {
    async fn complete(&self, request: CompletionRequest) -> Result<Completion, Error> {
        if request.cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let body = build_body(&request)?;
        let cancel = &request.cancel;
        let max_attempts = self.retries + 1;
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            match self.once(&body, cancel).await {
                Ok(completion) => return Ok(completion),
                Err(OnceError::Cancelled) => return Err(Error::Cancelled),
                Err(OnceError::Done(err)) => return Err(err),
                Err(OnceError::Retry(err)) => {
                    if attempt >= max_attempts {
                        return Err(err);
                    }
                    let delay = backoff_delay(attempt - 1);
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return Err(Error::Cancelled),
                        _ = tokio::time::sleep(delay) => {}
                    }
                }
            }
        }
    }
}

enum OnceError {
    Cancelled,
    /// Do not retry.
    Done(Error),
    /// `complete` retries this until the attempt budget is exhausted.
    Retry(Error),
}

/// Append `chat/completions` as one relative path.
///
/// `Url::join` drops the last segment when the base has no trailing slash, so
/// `https://host/v1` would become `https://host/chat/completions`. A base that
/// already ends in `/v1` must not gain a second `/v1` either.
fn chat_completions_url(base_url: &str) -> Result<Url, Error> {
    let mut url =
        Url::parse(base_url).map_err(|err| Error::Config(format!("llm.base_url: {err}")))?;
    match url.scheme() {
        "http" | "https" => {}
        other => {
            return Err(Error::Config(format!(
                "llm.base_url scheme {other} is not supported"
            )));
        }
    }
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|()| Error::Config("llm.base_url cannot be a base URL".to_string()))?;
        segments.pop_if_empty().push("chat").push("completions");
    }
    Ok(url)
}

fn build_body(request: &CompletionRequest) -> Result<Value, Error> {
    let mut body = json!({
        "model": request.model,
        "messages": request.messages.iter().map(message_to_json).collect::<Vec<_>>(),
        "max_tokens": request.max_output_tokens,
    });
    // An empty tools array plus tool_choice is a 400 on compatible servers,
    // and that 400 is not retried. Send both keys only when there is a tool.
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(request.tools.iter().map(tool_to_json).collect());
        body["tool_choice"] = tool_choice_to_json(&request.tool_choice);
    }
    if let Some(temperature) = request.temperature {
        if !temperature.is_finite() {
            return Err(llm_error(
                LlmErrorKind::BadRequest,
                "temperature must be finite",
            ));
        }
        let number = serde_json::Number::from_f64(f64::from(temperature))
            .ok_or_else(|| llm_error(LlmErrorKind::BadRequest, "temperature must be finite"))?;
        body["temperature"] = Value::Number(number);
    }
    Ok(body)
}

fn message_to_json(message: &Message) -> Value {
    match message {
        Message::System { content } => json!({ "role": "system", "content": content }),
        Message::User { content } => json!({ "role": "user", "content": content }),
        Message::Assistant {
            content,
            tool_calls,
        } => {
            let mut value = json!({
                "role": "assistant",
                "content": content,
            });
            if !tool_calls.is_empty() {
                value["tool_calls"] =
                    Value::Array(tool_calls.iter().map(tool_call_to_json).collect());
            }
            value
        }
        Message::Tool {
            tool_call_id,
            content,
        } => json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }),
    }
}

fn tool_call_to_json(call: &ToolCall) -> Value {
    json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments_raw,
        }
    })
}

fn tool_to_json(tool: &ToolDefinition) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    })
}

fn tool_choice_to_json(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => Value::String("auto".to_string()),
        ToolChoice::Function { name } => json!({
            "type": "function",
            "function": { "name": name }
        }),
    }
}

/// `true` when the status is retried (429 and 5xx).
fn classify_status(status: reqwest::StatusCode) -> (LlmErrorKind, bool) {
    match status.as_u16() {
        401 | 403 => (LlmErrorKind::Auth, false),
        429 => (LlmErrorKind::Unavailable, true),
        500..=599 => (LlmErrorKind::Unavailable, true),
        400..=499 => (LlmErrorKind::BadRequest, false),
        _ => (LlmErrorKind::Unavailable, false),
    }
}

fn transport_error(err: &reqwest::Error) -> OnceError {
    if err.is_timeout() {
        OnceError::Retry(llm_error(LlmErrorKind::Timeout, "request timed out"))
    } else if err.is_builder() {
        // Builder failures can quote a rejected header value. Do not forward them.
        OnceError::Done(llm_error(LlmErrorKind::BadRequest, "invalid HTTP request"))
    } else {
        OnceError::Retry(llm_error(
            LlmErrorKind::Unavailable,
            format!("request failed: {err}"),
        ))
    }
}

fn parse_completion(body: &str) -> Result<Completion, Error> {
    let value: Value = serde_json::from_str(body)
        .map_err(|_| llm_error(LlmErrorKind::Unavailable, "malformed chat completion"))?;
    let choices = value
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            llm_error(
                LlmErrorKind::Unavailable,
                "chat completion is missing choices",
            )
        })?;
    let choice = choices
        .first()
        .ok_or_else(|| llm_error(LlmErrorKind::Unavailable, "chat completion has no choices"))?;
    let message = choice.get("message").unwrap_or(&Value::Null);
    let content = message_content(message);
    let tool_calls = message_tool_calls(message);
    let finish_reason = choice.get("finish_reason").and_then(Value::as_str);
    let stop = map_stop(finish_reason, !tool_calls.is_empty());
    Ok(Completion {
        message: Message::Assistant {
            content,
            tool_calls,
        },
        stop,
    })
}

fn message_content(message: &Value) -> Option<String> {
    match message.get("content") {
        Some(Value::String(text)) => Some(text.clone()),
        // Some servers return content as an array of parts. Keep the text
        // instead of failing the completion.
        Some(other) if !other.is_null() => Some(compact_json(other)),
        _ => None,
    }
}

fn message_tool_calls(message: &Value) -> Vec<ToolCall> {
    let Some(items) = message.get("tool_calls").and_then(Value::as_array) else {
        return Vec::new();
    };
    items.iter().filter_map(parse_tool_call).collect()
}

fn parse_tool_call(value: &Value) -> Option<ToolCall> {
    if !value.is_object() {
        return None;
    }
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let function = value.get("function");
    let name = function
        .and_then(|item| item.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let arguments = function.and_then(|item| item.get("arguments"));
    let (arguments_raw, arguments) = split_arguments(arguments);
    Some(ToolCall {
        id,
        name,
        arguments_raw,
        arguments,
    })
}

/// A string is parsed. Invalid JSON keeps the original text and `arguments: None`.
/// An object (or any other JSON value) is stored as compact text and `Some`.
fn split_arguments(value: Option<&Value>) -> (String, Option<Value>) {
    match value {
        None => (String::new(), None),
        Some(Value::String(text)) => {
            let parsed = serde_json::from_str(text).ok();
            (text.clone(), parsed)
        }
        Some(other) => (compact_json(other), Some(other.clone())),
    }
}

fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// `length` and `content_filter` win even when tool calls are present.
/// The loop executes those calls, then appends a truncation note.
/// Any other finish reason follows the tool-call list: empty is `EndTurn`.
fn map_stop(finish_reason: Option<&str>, has_tool_calls: bool) -> StopReason {
    match finish_reason {
        Some("length") => StopReason::Length,
        Some("content_filter") => StopReason::ContentFilter,
        _ if has_tool_calls => StopReason::ToolUse,
        _ => StopReason::EndTurn,
    }
}

fn backoff_delay(retry_index: u32) -> Duration {
    let base = if retry_index == 0 {
        BACKOFF_FIRST_MS
    } else {
        BACKOFF_LATER_MS
    };
    let min = base.saturating_mul(8) / 10;
    let max = (base.saturating_mul(12) / 10).max(min);
    Duration::from_millis(rand::random_range(min..=max))
}

fn llm_error(kind: LlmErrorKind, message: impl Into<String>) -> Error {
    Error::Llm {
        kind,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_chat_completions_without_a_second_v1() {
        let cases = [
            (
                "https://api.openai.com/v1",
                "https://api.openai.com/v1/chat/completions",
            ),
            (
                "https://api.openai.com/v1/",
                "https://api.openai.com/v1/chat/completions",
            ),
            ("http://127.0.0.1:9", "http://127.0.0.1:9/chat/completions"),
            ("http://127.0.0.1:9/", "http://127.0.0.1:9/chat/completions"),
            ("http://[::1]:9/v1", "http://[::1]:9/v1/chat/completions"),
            (
                "https://example.com/openai/v1/",
                "https://example.com/openai/v1/chat/completions",
            ),
        ];
        for (base, expected) in cases {
            let url = chat_completions_url(base).unwrap();
            assert_eq!(url.as_str(), expected, "{base}");
        }
    }

    #[test]
    fn rejects_a_non_http_base_url() {
        let err = chat_completions_url("ftp://example.com/v1").unwrap_err();
        assert!(matches!(err, Error::Config(_)), "{err}");
        let err = OpenAiCompatible::new("not a url", "k", Duration::from_secs(1), 0).unwrap_err();
        assert!(matches!(err, Error::Config(_)), "{err}");
    }

    #[test]
    fn debug_redacts_the_api_key_and_caps_retries() {
        let secret = "sk-testsecretvalue";
        let llm = OpenAiCompatible::new(
            "https://api.openai.com/v1",
            secret,
            Duration::from_secs(12),
            99,
        )
        .unwrap();
        let debug = format!("{llm:?}");
        assert!(!debug.contains(secret), "debug output included the api key");
        assert!(debug.contains("[redacted]"));
        assert!(debug.contains("retries: 4"));
        assert!(!debug.contains("99"));
    }

    #[test]
    fn split_arguments_keeps_invalid_json_and_compacts_objects() {
        let (raw, parsed) = split_arguments(Some(&Value::String("not json".into())));
        assert_eq!(raw, "not json");
        assert_eq!(parsed, None);

        let original = " { \"path\" : \"a\" } ";
        let (raw, parsed) = split_arguments(Some(&Value::String(original.into())));
        assert_eq!(raw, original);
        assert_eq!(parsed, Some(json!({ "path": "a" })));

        let object = json!({"path": "src/lib.rs", "contents": "hi"});
        let (raw, parsed) = split_arguments(Some(&object));
        assert_eq!(parsed, Some(object.clone()));
        assert_eq!(raw, serde_json::to_string(&object).unwrap());
        assert!(!raw.contains(' '));

        let (raw, parsed) = split_arguments(None);
        assert_eq!(raw, "");
        assert_eq!(parsed, None);
    }

    #[test]
    fn stop_reason_prefers_length_and_content_filter() {
        assert_eq!(map_stop(Some("length"), true), StopReason::Length);
        assert_eq!(map_stop(Some("length"), false), StopReason::Length);
        assert_eq!(
            map_stop(Some("content_filter"), true),
            StopReason::ContentFilter
        );
        assert_eq!(
            map_stop(Some("content_filter"), false),
            StopReason::ContentFilter
        );
        assert_eq!(map_stop(Some("stop"), false), StopReason::EndTurn);
        assert_eq!(map_stop(None, false), StopReason::EndTurn);
        assert_eq!(map_stop(Some("stop"), true), StopReason::ToolUse);
        assert_eq!(map_stop(Some("tool_calls"), true), StopReason::ToolUse);
        assert_eq!(map_stop(Some("tool_calls"), false), StopReason::EndTurn);
        assert_eq!(map_stop(None, true), StopReason::ToolUse);
    }

    #[test]
    fn body_omits_temperature_and_sends_function_tools() {
        let request = CompletionRequest {
            model: "m".into(),
            messages: vec![
                Message::System {
                    content: "sys".into(),
                },
                Message::User {
                    content: "hi".into(),
                },
                Message::Assistant {
                    content: None,
                    tool_calls: vec![ToolCall {
                        id: "call_1".into(),
                        name: "read_file".into(),
                        arguments_raw: "{\"path\":\"a\"}".into(),
                        arguments: Some(json!({"path": "a"})),
                    }],
                },
                Message::Tool {
                    tool_call_id: "call_1".into(),
                    content: "ok".into(),
                },
            ],
            tools: vec![ToolDefinition {
                name: "read_file".into(),
                description: "Read".into(),
                parameters: json!({"type": "object"}),
            }],
            tool_choice: ToolChoice::Auto,
            max_output_tokens: 32,
            temperature: None,
            cancel: CancellationToken::new(),
        };
        let body = build_body(&request).unwrap();
        assert!(body.get("temperature").is_none());
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
        assert_eq!(body["tools"][0]["function"]["parameters"]["type"], "object");
        assert_eq!(body["messages"][2]["tool_calls"][0]["type"], "function");
        assert_eq!(
            body["messages"][2]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"a\"}"
        );
        assert!(body["messages"][2].get("tool_calls").is_some());
        assert_eq!(body["messages"][3]["role"], "tool");

        let mut forced = request;
        forced.tool_choice = ToolChoice::Function {
            name: "submit_plan".into(),
        };
        forced.temperature = Some(0.2);
        let body = build_body(&forced).unwrap();
        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["function"]["name"], "submit_plan");
        let temperature = body["temperature"].as_f64().unwrap();
        assert!((temperature - 0.2).abs() < 0.001);

        forced.temperature = Some(f32::NAN);
        let err = build_body(&forced).unwrap_err();
        assert!(matches!(
            err,
            Error::Llm {
                kind: LlmErrorKind::BadRequest,
                ..
            }
        ));
    }

    #[test]
    fn empty_tool_list_omits_tools_and_tool_choice() {
        let request = CompletionRequest {
            model: "m".into(),
            messages: vec![Message::User {
                content: "hi".into(),
            }],
            tools: Vec::new(),
            tool_choice: ToolChoice::Function {
                name: "submit_plan".into(),
            },
            max_output_tokens: 32,
            temperature: None,
            cancel: CancellationToken::new(),
        };
        let body = build_body(&request).unwrap();
        assert!(body.get("tools").is_none(), "{body}");
        assert!(body.get("tool_choice").is_none(), "{body}");
    }

    #[test]
    fn backoff_stays_within_twenty_percent() {
        for _ in 0..50 {
            let first = backoff_delay(0);
            assert!((400..=600).contains(&first.as_millis()), "{first:?}");
            let later = backoff_delay(3);
            assert!((1200..=1800).contains(&later.as_millis()), "{later:?}");
        }
    }

    #[test]
    fn parse_completion_does_not_panic_on_invalid_arguments() {
        let body = r#"{
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_bad",
                        "type": "function",
                        "function": { "name": "read_file", "arguments": "not-json" }
                    }]
                }
            }]
        }"#;
        let completion = parse_completion(body).unwrap();
        let Message::Assistant { tool_calls, .. } = completion.message else {
            panic!("expected assistant");
        };
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].arguments_raw, "not-json");
        assert_eq!(tool_calls[0].arguments, None);
        assert_eq!(completion.stop, StopReason::ToolUse);
    }
}
