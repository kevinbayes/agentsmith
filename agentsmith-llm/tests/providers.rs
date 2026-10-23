//! Anthropic and Gemini speak their own wire formats and the same retry table.

use std::collections::{BTreeMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agentsmith_core::{Error, LlmErrorKind};
use agentsmith_llm::{
    Anthropic, CompletionRequest, Gemini, Llm, Message, StopReason, ToolCall, ToolChoice,
    ToolDefinition,
};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::sync::CancellationToken;

const API_KEY: &str = "provider-secret";

#[tokio::test]
async fn anthropic_sends_messages_and_parses_tool_use() {
    let mock = Mock::start().await;
    mock.enqueue(
        200,
        json!({
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "reading"},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a"}}
            ]
        })
        .to_string(),
    );
    let llm = Anthropic::new(
        format!("http://{}/proxy", mock.addr),
        API_KEY,
        Duration::from_secs(5),
        0,
    )
    .unwrap();
    let completion = llm.complete(sample(ToolChoice::Auto, None)).await.unwrap();
    assert_eq!(completion.stop, StopReason::ToolUse);
    let Message::Assistant {
        content,
        tool_calls,
    } = completion.message
    else {
        panic!("assistant");
    };
    assert_eq!(content.as_deref(), Some("reading"));
    assert_eq!(tool_calls[0].id, "toolu_1");
    assert_eq!(tool_calls[0].name, "read_file");
    assert_eq!(tool_calls[0].arguments, Some(json!({"path": "a"})));

    let hit = &mock.hits()[0];
    assert_eq!(hit.path, "/proxy/v1/messages");
    assert_eq!(
        hit.headers.get("x-api-key").map(String::as_str),
        Some(API_KEY)
    );
    assert_eq!(
        hit.headers.get("anthropic-version").map(String::as_str),
        Some("2023-06-01")
    );
    assert!(hit.headers.get("authorization").is_none());
    assert!(!hit.body_raw.contains(API_KEY));
    assert_eq!(hit.body["model"], "claude-test");
    assert_eq!(hit.body["max_tokens"], 128);
    assert_eq!(hit.body["system"], "You are a test.");
    assert!(hit.body.get("temperature").is_none());
    assert_eq!(hit.body["tool_choice"]["type"], "auto");
    assert_eq!(hit.body["tools"][0]["name"], "read_file");
    assert_eq!(hit.body["tools"][0]["input_schema"]["type"], "object");
    assert_eq!(hit.body["messages"][0]["role"], "user");
    assert_eq!(
        hit.body["messages"][0]["content"][0]["text"],
        "Read the file."
    );
    assert_eq!(hit.body["messages"][1]["role"], "assistant");
    assert_eq!(hit.body["messages"][1]["content"][0]["type"], "tool_use");
    assert_eq!(hit.body["messages"][1]["content"][0]["id"], "call_prev");
    assert_eq!(
        hit.body["messages"][1]["content"][0]["input"]["path"],
        "src/lib.rs"
    );
    assert_eq!(hit.body["messages"][2]["role"], "user");
    assert_eq!(hit.body["messages"][2]["content"][0]["type"], "tool_result");
    assert_eq!(
        hit.body["messages"][2]["content"][0]["tool_use_id"],
        "call_prev"
    );
}

#[tokio::test]
async fn anthropic_forces_one_tool_and_retries_429() {
    let mock = Mock::start().await;
    mock.enqueue(429, "{}".into());
    mock.enqueue(
        200,
        json!({
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "ok"}]
        })
        .to_string(),
    );
    let llm = Anthropic::new(
        format!("http://{}", mock.addr),
        API_KEY,
        Duration::from_secs(5),
        1,
    )
    .unwrap();
    let completion = llm
        .complete(sample(
            ToolChoice::Function {
                name: "submit_plan".into(),
            },
            Some(0.2),
        ))
        .await
        .unwrap();
    assert_eq!(completion.stop, StopReason::EndTurn);
    let hits = mock.hits();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[1].body["tool_choice"]["type"], "tool");
    assert_eq!(hits[1].body["tool_choice"]["name"], "submit_plan");
    let temperature = hits[1].body["temperature"].as_f64().unwrap();
    assert!((temperature - 0.2).abs() < 0.001);
}

#[tokio::test]
async fn anthropic_maps_max_tokens_and_refusal() {
    let mock = Mock::start().await;
    mock.enqueue(
        200,
        json!({
            "stop_reason": "max_tokens",
            "content": [
                {"type": "text", "text": "cut"},
                {"type": "tool_use", "id": "t", "name": "read_file", "input": {}}
            ]
        })
        .to_string(),
    );
    mock.enqueue(
        200,
        json!({"stop_reason": "refusal", "content": [{"type": "text", "text": "no"}]}).to_string(),
    );
    let llm = Anthropic::new(
        format!("http://{}", mock.addr),
        API_KEY,
        Duration::from_secs(5),
        0,
    )
    .unwrap();
    assert_eq!(
        llm.complete(sample(ToolChoice::Auto, None))
            .await
            .unwrap()
            .stop,
        StopReason::Length
    );
    assert_eq!(
        llm.complete(sample(ToolChoice::Auto, None))
            .await
            .unwrap()
            .stop,
        StopReason::ContentFilter
    );
}

#[tokio::test]
async fn gemini_sends_function_calls_and_keeps_the_key_out_of_the_url() {
    let mock = Mock::start().await;
    mock.enqueue(
        200,
        json!({
            "candidates": [{
                "finishReason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "reading"},
                        {"functionCall": {"name": "read_file", "args": {"path": "a"}}}
                    ]
                }
            }]
        })
        .to_string(),
    );
    let llm = Gemini::new(
        format!("http://{}/", mock.addr),
        API_KEY,
        Duration::from_secs(5),
        0,
    )
    .unwrap();
    let mut request = sample(ToolChoice::Auto, None);
    request.model = "gemini-test".into();
    let completion = llm.complete(request).await.unwrap();
    assert_eq!(completion.stop, StopReason::ToolUse);
    let Message::Assistant {
        content,
        tool_calls,
    } = completion.message
    else {
        panic!("assistant");
    };
    assert_eq!(content.as_deref(), Some("reading"));
    assert_eq!(tool_calls[0].id, "call_1");
    assert_eq!(tool_calls[0].name, "read_file");
    assert_eq!(tool_calls[0].arguments, Some(json!({"path": "a"})));

    let hit = &mock.hits()[0];
    assert_eq!(hit.path, "/v1beta/models/gemini-test:generateContent");
    assert!(!hit.path.contains(API_KEY));
    assert_eq!(
        hit.headers.get("x-goog-api-key").map(String::as_str),
        Some(API_KEY)
    );
    assert!(hit.headers.get("authorization").is_none());
    assert!(!hit.body_raw.contains(API_KEY));
    assert_eq!(
        hit.body["systemInstruction"]["parts"][0]["text"],
        "You are a test."
    );
    assert_eq!(hit.body["generationConfig"]["maxOutputTokens"], 128);
    assert_eq!(hit.body["contents"][0]["role"], "user");
    assert_eq!(hit.body["contents"][1]["role"], "model");
    assert_eq!(
        hit.body["contents"][1]["parts"][0]["functionCall"]["name"],
        "read_file"
    );
    assert_eq!(hit.body["contents"][2]["role"], "user");
    assert_eq!(
        hit.body["contents"][2]["parts"][0]["functionResponse"]["name"],
        "read_file"
    );
    assert_eq!(
        hit.body["contents"][2]["parts"][0]["functionResponse"]["response"]["output"],
        "{\"ok\":true,\"content\":\"x\"}"
    );
    assert_eq!(
        hit.body["tools"][0]["functionDeclarations"][0]["name"],
        "read_file"
    );
    assert_eq!(
        hit.body["toolConfig"]["functionCallingConfig"]["mode"],
        "AUTO"
    );
}

#[tokio::test]
async fn gemini_any_mode_names_the_forced_tool_and_auth_is_not_retried() {
    let mock = Mock::start().await;
    mock.enqueue(401, "{}".into());
    let llm = Gemini::new(
        format!("http://{}", mock.addr),
        API_KEY,
        Duration::from_secs(5),
        2,
    )
    .unwrap();
    let mut request = sample(
        ToolChoice::Function {
            name: "submit_plan".into(),
        },
        Some(0.2),
    );
    request.model = "gemini-test".into();
    let err = llm.complete(request).await.unwrap_err();
    assert!(
        matches!(
            err,
            Error::Llm {
                kind: LlmErrorKind::Auth,
                ..
            }
        ),
        "{err}"
    );
    let hits = mock.hits();
    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].body["toolConfig"]["functionCallingConfig"]["mode"],
        "ANY"
    );
    assert_eq!(
        hits[0].body["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"][0],
        "submit_plan"
    );
}

#[tokio::test]
async fn gemini_safety_block_is_content_filter() {
    let mock = Mock::start().await;
    mock.enqueue(
        200,
        json!({"promptFeedback": {"blockReason": "SAFETY"}}).to_string(),
    );
    mock.enqueue(
        200,
        json!({
            "candidates": [{
                "finishReason": "MAX_TOKENS",
                "content": {"parts": [{"text": "cut"}]}
            }]
        })
        .to_string(),
    );
    let llm = Gemini::new(
        format!("http://{}", mock.addr),
        API_KEY,
        Duration::from_secs(5),
        0,
    )
    .unwrap();
    let mut blocked = sample(ToolChoice::Auto, None);
    blocked.model = "gemini-test".into();
    assert_eq!(
        llm.complete(blocked).await.unwrap().stop,
        StopReason::ContentFilter
    );
    let mut truncated = sample(ToolChoice::Auto, None);
    truncated.model = "gemini-test".into();
    assert_eq!(
        llm.complete(truncated).await.unwrap().stop,
        StopReason::Length
    );
}

fn sample(choice: ToolChoice, temperature: Option<f32>) -> CompletionRequest {
    CompletionRequest {
        model: "claude-test".into(),
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
            parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        }],
        tool_choice: choice,
        max_output_tokens: 128,
        temperature,
        cancel: CancellationToken::new(),
    }
}

#[derive(Clone)]
struct Hit {
    path: String,
    headers: BTreeMap<String, String>,
    body_raw: String,
    body: Value,
}

struct Scripted {
    status: u16,
    body: String,
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

    fn enqueue(&self, status: u16, body: String) {
        self.state
            .queue
            .lock()
            .expect("queue")
            .push_back(Scripted { status, body });
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
    };
    let header = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut lines = header.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("")
        .to_string();
    let mut content_length = 0usize;
    let mut headers = BTreeMap::new();
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim().to_string();
        if name == "content-length" {
            content_length = value.parse().unwrap_or(0);
        }
        headers.insert(name, value);
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
        path,
        headers,
        body_raw,
        body: parsed,
    });
    let Some(spec) = state.queue.lock().expect("queue").pop_front() else {
        return Ok(());
    };
    let header = format!(
        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n",
        status = spec.status,
        len = spec.body.len(),
    );
    socket.write_all(header.as_bytes()).await?;
    socket.write_all(spec.body.as_bytes()).await?;
    let _ = socket.shutdown().await;
    Ok(())
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
