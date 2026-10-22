use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agentsmith_core::{Error, LlmErrorKind};
use agentsmith_llm::{
    CompletionRequest, Llm, Message, OpenAiCompatible, StopReason, ToolCall, ToolChoice,
    ToolDefinition,
};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::sync::CancellationToken;

const API_KEY: &str = "test-key";

#[tokio::test]
async fn tool_call_round_trip_accepts_string_and_object_arguments() {
    let body = json!({
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call_str",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"src/lib.rs\"}"
                        }
                    },
                    {
                        "id": "call_obj",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "out.txt", "contents": "hi"}
                        }
                    }
                ]
            }
        }]
    });
    let mock = Mock::start().await;
    mock.enqueue(200, body.to_string(), Duration::ZERO);
    let llm = client(&mock, "/v1", 2, Duration::from_secs(5));

    let completion = llm
        .complete(sample_request(ToolChoice::Auto, None))
        .await
        .expect("complete");

    assert_eq!(completion.stop, StopReason::ToolUse);
    let Message::Assistant {
        content,
        tool_calls,
    } = completion.message
    else {
        panic!("expected assistant");
    };
    assert_eq!(content, None);
    assert_eq!(tool_calls.len(), 2);
    assert_eq!(tool_calls[0].id, "call_str");
    assert_eq!(tool_calls[0].name, "read_file");
    assert_eq!(tool_calls[0].arguments_raw, "{\"path\":\"src/lib.rs\"}");
    assert_eq!(tool_calls[0].arguments, Some(json!({"path": "src/lib.rs"})));
    assert_eq!(tool_calls[1].id, "call_obj");
    assert_eq!(tool_calls[1].name, "write_file");
    assert_eq!(
        tool_calls[1].arguments,
        Some(json!({"path": "out.txt", "contents": "hi"}))
    );
    assert_eq!(
        tool_calls[1].arguments_raw,
        serde_json::to_string(&tool_calls[1].arguments).unwrap()
    );
    assert!(!tool_calls[1].arguments_raw.contains(' '));

    let hits = mock.hits();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].method, "POST");
    assert_eq!(hits[0].path, "/v1/chat/completions");
    assert_eq!(hits[0].authorization, format!("Bearer {API_KEY}"));
    assert!(!hits[0].body_raw.contains(API_KEY));
    let sent = &hits[0].body;
    assert_eq!(sent["model"], "gpt-test");
    assert_eq!(sent["max_tokens"], 128);
    assert!(sent.get("temperature").is_none());
    assert_eq!(sent["tool_choice"], "auto");
    assert_eq!(sent["tools"][0]["type"], "function");
    assert_eq!(sent["tools"][0]["function"]["name"], "read_file");
    assert_eq!(
        sent["tools"][0]["function"]["description"],
        "Read a UTF-8 file"
    );
    assert_eq!(sent["tools"][0]["function"]["parameters"]["type"], "object");
    assert_eq!(sent["messages"][0]["role"], "system");
    assert_eq!(sent["messages"][1]["role"], "user");
    assert_eq!(sent["messages"][2]["role"], "assistant");
    assert_eq!(sent["messages"][2]["content"], Value::Null);
    assert_eq!(
        sent["messages"][2]["tool_calls"][0]["function"]["arguments"],
        "{\"path\":\"src/lib.rs\"}"
    );
    assert_eq!(sent["messages"][2]["tool_calls"][0]["type"], "function");
    assert_eq!(sent["messages"][3]["role"], "tool");
    assert_eq!(sent["messages"][3]["tool_call_id"], "call_prev");
}

#[tokio::test]
async fn trailing_slash_base_and_function_tool_choice() {
    let mock = Mock::start().await;
    mock.enqueue(200, end_turn_body(), Duration::ZERO);
    let llm = client(&mock, "/v1/", 0, Duration::from_secs(5));
    let mut request = sample_request(
        ToolChoice::Function {
            name: "submit_plan".into(),
        },
        Some(0.2),
    );
    request.tools.clear();
    request.messages.truncate(2);

    let completion = llm.complete(request).await.expect("complete");
    assert_eq!(completion.stop, StopReason::EndTurn);

    let hit = &mock.hits()[0];
    assert_eq!(hit.path, "/v1/chat/completions");
    assert_eq!(hit.body["tool_choice"]["type"], "function");
    assert_eq!(hit.body["tool_choice"]["function"]["name"], "submit_plan");
    let temperature = hit.body["temperature"].as_f64().unwrap();
    assert!((temperature - 0.2).abs() < 0.001);
}

#[tokio::test]
async fn invalid_arguments_string_returns_ok_with_none() {
    let body = json!({
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "not-json {"
                    }
                }]
            }
        }]
    });
    let mock = Mock::start().await;
    mock.enqueue(200, body.to_string(), Duration::ZERO);
    let llm = client(&mock, "/v1", 2, Duration::from_secs(5));

    let completion = llm
        .complete(sample_request(ToolChoice::Auto, None))
        .await
        .expect("invalid arguments must not fail complete");

    assert_eq!(completion.stop, StopReason::ToolUse);
    let Message::Assistant { tool_calls, .. } = completion.message else {
        panic!("expected assistant");
    };
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].arguments_raw, "not-json {");
    assert_eq!(tool_calls[0].arguments, None);
    assert_eq!(mock.hits().len(), 1);
}

#[tokio::test]
async fn auth_and_bad_request_are_not_retried() {
    for (status, kind) in [
        (401, LlmErrorKind::Auth),
        (403, LlmErrorKind::Auth),
        (400, LlmErrorKind::BadRequest),
        (404, LlmErrorKind::BadRequest),
    ] {
        let mock = Mock::start().await;
        let leaked = format!("prompt text and {API_KEY} and sk-testsecretvalue");
        mock.enqueue(status, leaked.clone(), Duration::ZERO);
        let llm = client(&mock, "/v1", 2, Duration::from_secs(5));
        let err = llm
            .complete(sample_request(ToolChoice::Auto, None))
            .await
            .expect_err("status {status}");
        match err {
            Error::Llm { kind: got, .. } => assert_eq!(got, kind, "status {status}"),
            other => panic!("status {status}: expected llm error, got {other}"),
        }
        assert_eq!(mock.hits().len(), 1, "status {status}");
    }
}

#[tokio::test]
async fn error_text_does_not_include_the_body_or_the_key() {
    let mock = Mock::start().await;
    let leaked = format!("prompt text and {API_KEY} and sk-testsecretvalue");
    mock.enqueue(401, leaked, Duration::ZERO);
    let llm = client(&mock, "/v1", 2, Duration::from_secs(5));
    let err = llm
        .complete(sample_request(ToolChoice::Auto, None))
        .await
        .expect_err("401");
    let rendered = err.to_string();
    assert!(matches!(
        err,
        Error::Llm {
            kind: LlmErrorKind::Auth,
            ..
        }
    ));
    assert!(!rendered.contains(API_KEY), "{rendered}");
    assert!(!rendered.contains("sk-testsecretvalue"), "{rendered}");
    assert!(!rendered.contains("prompt text"), "{rendered}");
    assert!(rendered.contains("401"), "{rendered}");
    assert_eq!(mock.hits().len(), 1);
}

#[tokio::test]
async fn too_many_requests_is_retried_then_unavailable() {
    let mock = Mock::start().await;
    for _ in 0..3 {
        mock.enqueue(429, "slow down".into(), Duration::ZERO);
    }
    let llm = client(&mock, "/v1", 2, Duration::from_secs(5));
    let started = Instant::now();
    let err = llm
        .complete(sample_request(ToolChoice::Auto, None))
        .await
        .expect_err("429");
    let elapsed = started.elapsed();
    assert!(matches!(
        err,
        Error::Llm {
            kind: LlmErrorKind::Unavailable,
            ..
        }
    ));
    assert_eq!(mock.hits().len(), 3);
    // Two waits: 500ms and 1500ms, each at least 80% after jitter.
    assert!(
        elapsed >= Duration::from_millis(1400),
        "backoff was shorter than 500ms then 1500ms: {elapsed:?}"
    );
}

#[tokio::test]
async fn server_error_is_retried_then_unavailable() {
    let mock = Mock::start().await;
    mock.enqueue(500, "boom".into(), Duration::ZERO);
    mock.enqueue(502, "boom".into(), Duration::ZERO);
    let llm = client(&mock, "/v1", 1, Duration::from_secs(5));
    let started = Instant::now();
    let err = llm
        .complete(sample_request(ToolChoice::Auto, None))
        .await
        .expect_err("500");
    assert!(matches!(
        err,
        Error::Llm {
            kind: LlmErrorKind::Unavailable,
            ..
        }
    ));
    assert_eq!(mock.hits().len(), 2);
    assert!(
        started.elapsed() >= Duration::from_millis(350),
        "5xx was not retried with backoff"
    );
}

#[tokio::test]
async fn retry_stops_when_a_later_attempt_succeeds() {
    let mock = Mock::start().await;
    mock.enqueue(429, "slow down".into(), Duration::ZERO);
    mock.enqueue(200, end_turn_body(), Duration::ZERO);
    let llm = client(&mock, "/v1", 2, Duration::from_secs(5));
    let completion = llm
        .complete(sample_request(ToolChoice::Auto, None))
        .await
        .expect("second attempt should succeed");
    assert_eq!(completion.stop, StopReason::EndTurn);
    assert_eq!(mock.hits().len(), 2);
}

#[tokio::test]
async fn timeout_is_retried_then_timeout() {
    let mock = Mock::start().await;
    mock.enqueue(200, end_turn_body(), Duration::from_secs(30));
    mock.enqueue(200, end_turn_body(), Duration::from_secs(30));
    let llm = client(&mock, "/v1", 1, Duration::from_millis(200));
    let result = tokio::time::timeout(
        Duration::from_secs(3),
        llm.complete(sample_request(ToolChoice::Auto, None)),
    )
    .await
    .expect("timeout retries should finish before the server delay");
    let err = result.expect_err("timeout");
    assert!(
        matches!(
            err,
            Error::Llm {
                kind: LlmErrorKind::Timeout,
                ..
            }
        ),
        "{err}"
    );
    assert_eq!(mock.hits().len(), 2);
}

#[tokio::test]
async fn cancel_during_the_request_returns_cancelled() {
    let mock = Mock::start().await;
    mock.enqueue(200, end_turn_body(), Duration::from_secs(30));
    let llm = client(&mock, "/v1", 2, Duration::from_secs(30));
    let cancel = CancellationToken::new();
    let cancel_task = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(40)).await;
        cancel_task.cancel();
    });
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        llm.complete(sample_request_with(cancel, ToolChoice::Auto, None)),
    )
    .await
    .expect("cancel should abort the in-flight request");
    assert!(matches!(result, Err(Error::Cancelled)));
}

#[tokio::test]
async fn cancel_during_backoff_does_not_send_another_request() {
    let mock = Mock::start().await;
    mock.enqueue(429, "slow down".into(), Duration::ZERO);
    mock.enqueue(429, "slow down".into(), Duration::ZERO);
    let llm = client(&mock, "/v1", 2, Duration::from_secs(5));
    let cancel = CancellationToken::new();
    let cancel_task = cancel.clone();
    let hits = Arc::clone(&mock.state);
    tokio::spawn(async move {
        loop {
            if !hits.hits.lock().expect("hits").is_empty() {
                cancel_task.cancel();
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });
    let err = tokio::time::timeout(
        Duration::from_secs(2),
        llm.complete(sample_request_with(cancel, ToolChoice::Auto, None)),
    )
    .await
    .expect("cancel during backoff")
    .expect_err("cancelled");
    assert!(matches!(err, Error::Cancelled), "{err}");
    assert_eq!(mock.hits().len(), 1);
}

#[tokio::test]
async fn already_cancelled_does_not_send() {
    let mock = Mock::start().await;
    mock.enqueue(200, end_turn_body(), Duration::ZERO);
    let llm = client(&mock, "/v1", 2, Duration::from_secs(5));
    let cancel = CancellationToken::new();
    cancel.cancel();
    let err = llm
        .complete(sample_request_with(cancel, ToolChoice::Auto, None))
        .await
        .expect_err("cancelled");
    assert!(matches!(err, Error::Cancelled));
    assert!(mock.hits().is_empty());
}

#[tokio::test]
async fn finish_reasons_map_length_and_content_filter_with_tool_calls() {
    let cases = [
        (Some("length"), true, StopReason::Length),
        (Some("content_filter"), true, StopReason::ContentFilter),
        (Some("length"), false, StopReason::Length),
        (Some("content_filter"), false, StopReason::ContentFilter),
        (Some("stop"), false, StopReason::EndTurn),
        (Some("stop"), true, StopReason::ToolUse),
        (Some("tool_calls"), true, StopReason::ToolUse),
        (Some("tool_calls"), false, StopReason::EndTurn),
        (None, false, StopReason::EndTurn),
        (None, true, StopReason::ToolUse),
    ];
    let mock = Mock::start().await;
    for (finish, with_tools, _) in &cases {
        mock.enqueue(200, finish_body(*finish, *with_tools), Duration::ZERO);
    }
    let llm = client(&mock, "/v1", 0, Duration::from_secs(5));
    for (finish, with_tools, expected) in cases {
        let completion = llm
            .complete(sample_request(ToolChoice::Auto, None))
            .await
            .expect("complete");
        assert_eq!(
            completion.stop, expected,
            "finish={finish:?} tools={with_tools}"
        );
        let Message::Assistant {
            content,
            tool_calls,
        } = completion.message
        else {
            panic!("expected assistant");
        };
        if with_tools {
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].arguments, Some(json!({"path": "a"})));
        } else {
            assert!(tool_calls.is_empty());
            assert_eq!(content.as_deref(), Some("hello"));
        }
    }
}

fn client(mock: &Mock, prefix: &str, retries: u32, timeout: Duration) -> OpenAiCompatible {
    let base = format!("http://{}{prefix}", mock.addr);
    OpenAiCompatible::new(base, API_KEY, timeout, retries).expect("client")
}

fn sample_request(choice: ToolChoice, temperature: Option<f32>) -> CompletionRequest {
    sample_request_with(CancellationToken::new(), choice, temperature)
}

fn sample_request_with(
    cancel: CancellationToken,
    choice: ToolChoice,
    temperature: Option<f32>,
) -> CompletionRequest {
    CompletionRequest {
        model: "gpt-test".into(),
        messages: vec![
            Message::System {
                content: "You are a test.".into(),
            },
            Message::User {
                content: "Read the file.".into(),
            },
            Message::Assistant {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_prev".into(),
                    name: "read_file".into(),
                    arguments_raw: "{\"path\":\"src/lib.rs\"}".into(),
                    arguments: Some(json!({"path": "src/lib.rs"})),
                }],
            },
            Message::Tool {
                tool_call_id: "call_prev".into(),
                content: "{\"ok\":true,\"content\":\"x\"}".into(),
            },
        ],
        tools: vec![ToolDefinition {
            name: "read_file".into(),
            description: "Read a UTF-8 file".into(),
            parameters: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}}
            }),
        }],
        tool_choice: choice,
        max_output_tokens: 128,
        temperature,
        cancel,
    }
}

fn end_turn_body() -> String {
    finish_body(Some("stop"), false)
}

fn finish_body(finish: Option<&str>, with_tools: bool) -> String {
    let mut message = json!({
        "role": "assistant",
        "content": if with_tools { Value::Null } else { json!("hello") },
    });
    if with_tools {
        message["tool_calls"] = json!([{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": "{\"path\":\"a\"}"
            }
        }]);
    }
    let mut choice = json!({ "message": message });
    if let Some(finish) = finish {
        choice["finish_reason"] = json!(finish);
    }
    json!({ "choices": [choice] }).to_string()
}

#[derive(Clone)]
struct Hit {
    method: String,
    path: String,
    authorization: String,
    body_raw: String,
    body: Value,
}

struct Scripted {
    status: u16,
    body: String,
    delay: Duration,
}

struct State {
    queue: Mutex<VecDeque<Scripted>>,
    hits: Mutex<Vec<Hit>>,
}

struct Mock {
    addr: SocketAddr,
    state: Arc<State>,
}

impl Mock {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let state = Arc::new(State {
            queue: Mutex::new(VecDeque::new()),
            hits: Mutex::new(Vec::new()),
        });
        let serve = Arc::clone(&state);
        tokio::spawn(async move {
            while let Ok((socket, _)) = listener.accept().await {
                let state = Arc::clone(&serve);
                tokio::spawn(async move {
                    let _ = handle_conn(socket, state).await;
                });
            }
        });
        Self { addr, state }
    }

    fn enqueue(&self, status: u16, body: String, delay: Duration) {
        self.state.queue.lock().expect("queue").push_back(Scripted {
            status,
            body,
            delay,
        });
    }

    fn hits(&self) -> Vec<Hit> {
        self.state.hits.lock().expect("hits").clone()
    }
}

async fn handle_conn(mut socket: TcpStream, state: Arc<State>) -> std::io::Result<()> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_bytes(&buf, b"\r\n\r\n") {
            break pos;
        }
        if buf.len() > 64 * 1024 {
            return Ok(());
        }
    };
    let header = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut lines = header.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut content_length = 0usize;
    let mut authorization = String::new();
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("content-length") {
            content_length = value.trim().parse().unwrap_or(0);
        } else if name.eq_ignore_ascii_case("authorization") {
            authorization = value.trim().to_string();
        }
    }
    let mut body = buf[header_end + 4..].to_vec();
    while body.len() < content_length {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(content_length);
    let body_raw = String::from_utf8_lossy(&body).into_owned();
    let parsed = serde_json::from_slice(&body).unwrap_or(Value::Null);
    state.hits.lock().expect("hits").push(Hit {
        method,
        path,
        authorization,
        body_raw,
        body: parsed,
    });

    let spec = state.queue.lock().expect("queue").pop_front();
    let Some(spec) = spec else {
        return Ok(());
    };
    if !spec.delay.is_zero() {
        tokio::time::sleep(spec.delay).await;
    }
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n",
        status = spec.status,
        reason = reason_phrase(spec.status),
        len = spec.body.len(),
    );
    socket.write_all(header.as_bytes()).await?;
    socket.write_all(spec.body.as_bytes()).await?;
    let _ = socket.shutdown().await;
    Ok(())
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        _ => "Error",
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
