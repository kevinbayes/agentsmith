use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use agentsmith_core::config::Config;
use agentsmith_core::{Error, LlmErrorKind, ToolErrorKind};
use agentsmith_llm::{Completion, CompletionRequest, Llm, Message, StopReason, ToolCall};
use agentsmith_runtime::{
    run_foreground, AgentSpec, CancellationToken, ForegroundParams, LoopLimits, Phase, ReactLoop,
    RunContext, RunOutcome, RunStatus, RunTools, Supervisor, Transcript,
};
use agentsmith_tools::{Tool, ToolCtx, ToolDefinition, ToolOutput, ToolRegistry};
use serde_json::{json, Value};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}

struct Seen {
    model: String,
    tool_choice_auto: bool,
    tool_names: Vec<String>,
    messages: Vec<Message>,
}

enum Step {
    Ok(Completion),
    Err(Error),
    Cancel(Completion),
}

struct FakeLlm {
    steps: Mutex<VecDeque<Step>>,
    seen: Mutex<Vec<Seen>>,
}

impl FakeLlm {
    fn new(steps: Vec<Step>) -> Arc<Self> {
        Arc::new(Self {
            steps: Mutex::new(steps.into()),
            seen: Mutex::new(Vec::new()),
        })
    }

    fn seen(&self) -> Vec<Seen> {
        lock(&self.seen).clone()
    }
}

impl Clone for Seen {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            tool_choice_auto: self.tool_choice_auto,
            tool_names: self.tool_names.clone(),
            messages: self.messages.clone(),
        }
    }
}

#[async_trait::async_trait]
impl Llm for FakeLlm {
    async fn complete(&self, request: CompletionRequest) -> Result<Completion, Error> {
        lock(&self.seen).push(Seen {
            model: request.model.clone(),
            tool_choice_auto: matches!(request.tool_choice, agentsmith_llm::ToolChoice::Auto),
            tool_names: request.tools.iter().map(|tool| tool.name.clone()).collect(),
            messages: request.messages.clone(),
        });
        let step = lock(&self.steps)
            .pop_front()
            .expect("complete called more times than scripted");
        match step {
            Step::Ok(completion) => Ok(completion),
            Step::Err(err) => Err(err),
            Step::Cancel(completion) => {
                request.cancel.cancel();
                Ok(completion)
            }
        }
    }
}

struct ScriptedTool {
    name: String,
    calls: Arc<AtomicUsize>,
    results: Mutex<VecDeque<Result<String, Error>>>,
}

impl ScriptedTool {
    fn new(name: &str, results: Vec<Result<String, Error>>) -> (Arc<Self>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let tool = Arc::new(Self {
            name: name.to_string(),
            calls: Arc::clone(&calls),
            results: Mutex::new(results.into()),
        });
        (tool, calls)
    }
}

#[async_trait::async_trait]
impl Tool for ScriptedTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.clone(),
            description: self.name.clone(),
            parameters: json!({"type": "object", "additionalProperties": false}),
        }
    }

    async fn call(&self, _ctx: ToolCtx, _arguments: Value) -> Result<ToolOutput, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match lock(&self.results).pop_front() {
            Some(Ok(content)) => Ok(ToolOutput { content }),
            Some(Err(err)) => Err(err),
            None => Err(Error::Tool {
                tool: self.name.clone(),
                kind: ToolErrorKind::Execution,
                message: "no scripted result".into(),
            }),
        }
    }
}

fn limits(max_iters: u32, max_errors: u32) -> LoopLimits {
    LoopLimits {
        react_max_iterations: max_iters,
        max_consecutive_tool_errors: max_errors,
    }
}

fn tool_call(id: &str, name: &str, arguments: Option<Value>) -> ToolCall {
    let arguments_raw = match &arguments {
        Some(value) => serde_json::to_string(value).unwrap(),
        None => "not-json".into(),
    };
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments_raw,
        arguments,
    }
}

fn assistant(content: Option<&str>, calls: Vec<ToolCall>, stop: StopReason) -> Completion {
    Completion {
        message: Message::Assistant {
            content: content.map(str::to_string),
            tool_calls: calls,
        },
        stop,
    }
}

fn execution(tool: &str, message: &str) -> Error {
    Error::Tool {
        tool: tool.into(),
        kind: ToolErrorKind::Execution,
        message: message.into(),
    }
}

struct Harness {
    llm: Arc<FakeLlm>,
    supervisor: Arc<Supervisor>,
    run_id: uuid::Uuid,
    ctx: RunContext,
}

fn harness(
    steps: Vec<Step>,
    tools: Vec<Arc<dyn Tool>>,
    visible: Option<Vec<String>>,
    max_iters: u32,
    max_errors: u32,
    cancel: CancellationToken,
) -> Harness {
    let llm = FakeLlm::new(steps);
    let supervisor = Arc::new(Supervisor::new());
    let run_id = agentsmith_core::id::new_id();
    let registry = ToolRegistry::new();
    for tool in tools {
        registry.register(tool).unwrap();
    }
    let ctx = RunContext {
        run_id,
        spec: AgentSpec::react_default(),
        task: "ship the change".into(),
        extra_context: None,
        model: "spec-model".into(),
        max_output_tokens: 64,
        temperature: None,
        max_output_bytes: 32 * 1024,
        workdir: std::env::temp_dir(),
        llm: llm.clone(),
        tools: RunTools::resolve(registry, None, visible.as_deref()).unwrap(),
        limits: limits(max_iters, max_errors),
        cancel,
        events: supervisor.clone(),
        transcript: Transcript::empty(),
    };
    Harness {
        llm,
        supervisor,
        run_id,
        ctx,
    }
}

struct Done {
    outcome: RunOutcome,
    llm: Arc<FakeLlm>,
    supervisor: Arc<Supervisor>,
}

async fn run(harness: Harness) -> Done {
    let outcome = ReactLoop.run(harness.ctx).await.expect("run");
    Done {
        outcome,
        llm: harness.llm,
        supervisor: harness.supervisor,
    }
}

fn tool_json(transcript: &[Message], id: &str) -> Value {
    transcript
        .iter()
        .find_map(|message| match message {
            Message::Tool {
                tool_call_id,
                content,
            } if tool_call_id == id => Some(serde_json::from_str(content).unwrap()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing tool message {id}"))
}

fn user_text(transcript: &[Message]) -> Vec<&str> {
    transcript
        .iter()
        .filter_map(|message| match message {
            Message::User { content } => Some(content.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn end_turn_without_tool_calls_succeeds() {
    let harness = harness(
        vec![Step::Ok(assistant(
            Some("hello"),
            vec![],
            StopReason::EndTurn,
        ))],
        Vec::new(),
        None,
        4,
        5,
        CancellationToken::new(),
    );
    let id = harness.run_id;
    let done = run(harness).await;
    let outcome = done.outcome;
    assert_eq!(outcome.status, RunStatus::Succeeded);
    assert_eq!(outcome.summary, "hello");
    assert!(outcome.error_kind.is_none());
    let seen = done.llm.seen();
    assert_eq!(seen.len(), 1);
    assert!(seen[0].tool_choice_auto);
    assert_eq!(seen[0].model, "spec-model");
    assert_eq!(
        seen[0].messages[0],
        Message::System {
            content: AgentSpec::react_default().system_prompt,
        }
    );
    let snap = done.supervisor.snapshot(id).unwrap();
    assert_eq!(snap.status, RunStatus::Succeeded);
    assert_eq!(snap.phase, Phase::React);
    assert_eq!(snap.iteration, 0);
    assert!(snap.error_kind.is_none());
}

#[tokio::test]
async fn empty_tool_use_is_success() {
    let harness = harness(
        vec![Step::Ok(assistant(
            Some("via tool use"),
            vec![],
            StopReason::ToolUse,
        ))],
        Vec::new(),
        None,
        4,
        5,
        CancellationToken::new(),
    );
    let outcome = run(harness).await.outcome;
    assert_eq!(outcome.status, RunStatus::Succeeded);
    assert_eq!(outcome.summary, "via tool use");
    assert!(outcome.error_kind.is_none());
}

#[tokio::test]
async fn length_without_tool_calls_fails() {
    let harness = harness(
        vec![Step::Ok(assistant(
            Some("partial"),
            vec![],
            StopReason::Length,
        ))],
        Vec::new(),
        None,
        4,
        5,
        CancellationToken::new(),
    );
    let outcome = run(harness).await.outcome;
    assert_eq!(outcome.status, RunStatus::Failed);
    assert_eq!(outcome.error_kind.as_deref(), Some("length"));
    assert_eq!(outcome.summary, "partial");
}

#[tokio::test]
async fn content_filter_without_tool_calls_fails() {
    let harness = harness(
        vec![Step::Ok(assistant(None, vec![], StopReason::ContentFilter))],
        Vec::new(),
        None,
        4,
        5,
        CancellationToken::new(),
    );
    let outcome = run(harness).await.outcome;
    assert_eq!(outcome.status, RunStatus::Failed);
    assert_eq!(outcome.error_kind.as_deref(), Some("content_filter"));
    assert_eq!(outcome.summary, "content_filter");
}

async fn truncated_tool_then_success(stop_reason: StopReason, note: &str) {
    let (tool, calls) = ScriptedTool::new("echo", vec![Ok("echoed".into())]);
    let harness = harness(
        vec![
            Step::Ok(assistant(
                Some("partial"),
                vec![tool_call("c1", "echo", Some(json!({})))],
                stop_reason,
            )),
            Step::Ok(assistant(Some("done"), vec![], StopReason::EndTurn)),
        ],
        vec![tool],
        None,
        4,
        5,
        CancellationToken::new(),
    );
    let done = run(harness).await;
    let outcome = done.outcome;
    assert_eq!(outcome.status, RunStatus::Succeeded);
    assert_eq!(outcome.summary, "done");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        tool_json(&outcome.transcript, "c1")["content"],
        json!("echoed")
    );
    let users = user_text(&outcome.transcript);
    assert_eq!(*users.last().unwrap(), note);
    let seen = done.llm.seen();
    assert_eq!(seen.len(), 2);
    let second = &seen[1].messages;
    assert!(matches!(
        second.last(),
        Some(Message::User { content }) if content == note
    ));
}

#[tokio::test]
async fn length_with_a_tool_call_appends_the_truncation_note() {
    truncated_tool_then_success(
        StopReason::Length,
        "The previous assistant message was truncated (finish_reason=length).",
    )
    .await;
}

#[tokio::test]
async fn content_filter_with_a_tool_call_appends_the_truncation_note() {
    truncated_tool_then_success(
        StopReason::ContentFilter,
        "The previous assistant message was truncated (finish_reason=content_filter).",
    )
    .await;
}

#[tokio::test]
async fn tool_error_is_an_observation_and_a_later_turn_can_succeed() {
    let (tool, calls) = ScriptedTool::new("echo", vec![Err(execution("echo", "disk full"))]);
    let harness = harness(
        vec![
            Step::Ok(assistant(
                None,
                vec![tool_call("c1", "echo", Some(json!({})))],
                StopReason::ToolUse,
            )),
            Step::Ok(assistant(Some("recovered"), vec![], StopReason::EndTurn)),
        ],
        vec![tool],
        None,
        4,
        5,
        CancellationToken::new(),
    );
    let outcome = run(harness).await.outcome;
    assert_eq!(outcome.status, RunStatus::Succeeded);
    assert_eq!(outcome.summary, "recovered");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let obs = tool_json(&outcome.transcript, "c1");
    assert_eq!(obs["ok"], json!(false));
    assert_eq!(obs["error"]["kind"], json!("execution"));
    assert_eq!(obs["error"]["message"], json!("disk full"));
}

#[tokio::test]
async fn consecutive_tool_errors_fail_the_run_before_later_calls() {
    let (tool, calls) = ScriptedTool::new(
        "echo",
        vec![
            Err(execution("echo", "one")),
            Err(execution("echo", "two")),
            Err(execution("echo", "three")),
        ],
    );
    let harness = harness(
        vec![Step::Ok(assistant(
            None,
            vec![
                tool_call("c1", "echo", Some(json!({}))),
                tool_call("c2", "echo", Some(json!({}))),
                tool_call("c3", "echo", Some(json!({}))),
            ],
            StopReason::ToolUse,
        ))],
        vec![tool],
        None,
        4,
        2,
        CancellationToken::new(),
    );
    let done = run(harness).await;
    let outcome = done.outcome;
    assert_eq!(outcome.status, RunStatus::Failed);
    assert_eq!(outcome.error_kind.as_deref(), Some("tool_errors"));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(outcome.transcript.iter().all(
        |message| !matches!(message, Message::Tool { tool_call_id, .. } if tool_call_id == "c3")
    ));
    assert_eq!(done.llm.seen().len(), 1);
}

#[tokio::test]
async fn a_successful_tool_resets_the_consecutive_error_counter() {
    let (tool, calls) = ScriptedTool::new(
        "echo",
        vec![
            Err(execution("echo", "one")),
            Ok("ok".into()),
            Err(execution("echo", "two")),
        ],
    );
    let harness = harness(
        vec![
            Step::Ok(assistant(
                None,
                vec![
                    tool_call("c1", "echo", Some(json!({}))),
                    tool_call("c2", "echo", Some(json!({}))),
                    tool_call("c3", "echo", Some(json!({}))),
                ],
                StopReason::ToolUse,
            )),
            Step::Ok(assistant(Some("done"), vec![], StopReason::EndTurn)),
        ],
        vec![tool],
        None,
        4,
        2,
        CancellationToken::new(),
    );
    let done = run(harness).await;
    let outcome = done.outcome;
    assert_eq!(
        outcome.status,
        RunStatus::Succeeded,
        "{:?}",
        outcome.error_kind
    );
    assert_eq!(outcome.summary, "done");
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(done.llm.seen().len(), 2);
}

#[tokio::test]
async fn a_zero_error_cap_spares_success_and_stops_on_the_first_error() {
    let (tool, calls) = ScriptedTool::new("echo", vec![Ok("ok".into())]);
    let outcome = run(harness(
        vec![
            Step::Ok(assistant(
                None,
                vec![tool_call("c1", "echo", Some(json!({})))],
                StopReason::ToolUse,
            )),
            Step::Ok(assistant(Some("done"), vec![], StopReason::EndTurn)),
        ],
        vec![tool],
        None,
        4,
        0,
        CancellationToken::new(),
    ))
    .await
    .outcome;
    assert_eq!(outcome.status, RunStatus::Succeeded);
    assert_eq!(outcome.summary, "done");
    assert!(outcome.error_kind.is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let (tool, calls) = ScriptedTool::new("echo", vec![Err(execution("echo", "disk full"))]);
    let outcome = run(harness(
        vec![Step::Ok(assistant(
            None,
            vec![tool_call("c1", "echo", Some(json!({})))],
            StopReason::ToolUse,
        ))],
        vec![tool],
        None,
        4,
        0,
        CancellationToken::new(),
    ))
    .await
    .outcome;
    assert_eq!(outcome.status, RunStatus::Failed);
    assert_eq!(outcome.error_kind.as_deref(), Some("tool_errors"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(tool_json(&outcome.transcript, "c1")["ok"], json!(false));
}

#[tokio::test]
async fn a_fired_cancel_token_wins_over_a_zero_iteration_cap() {
    let cancel = CancellationToken::new();
    cancel.cancel();
    let done = run(harness(vec![], Vec::new(), None, 0, 5, cancel)).await;
    assert_eq!(done.outcome.status, RunStatus::Cancelled);
    assert!(done.outcome.error_kind.is_none());
    assert!(done.llm.seen().is_empty());
}

#[tokio::test]
async fn none_arguments_do_not_invoke_the_tool() {
    let (tool, calls) = ScriptedTool::new("echo", vec![Ok("nope".into())]);
    let harness = harness(
        vec![
            Step::Ok(assistant(
                None,
                vec![tool_call("c1", "echo", None)],
                StopReason::ToolUse,
            )),
            Step::Ok(assistant(Some("after"), vec![], StopReason::EndTurn)),
        ],
        vec![tool],
        None,
        4,
        5,
        CancellationToken::new(),
    );
    let outcome = run(harness).await.outcome;
    assert_eq!(outcome.status, RunStatus::Succeeded);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let obs = tool_json(&outcome.transcript, "c1");
    assert_eq!(obs["ok"], json!(false));
    assert_eq!(obs["error"]["kind"], json!("invalid_args"));
}

#[tokio::test]
async fn max_iterations_stops_after_the_last_observation() {
    let (tool, calls) = ScriptedTool::new("echo", vec![Ok("once".into())]);
    let harness = harness(
        vec![Step::Ok(assistant(
            None,
            vec![tool_call("c1", "echo", Some(json!({})))],
            StopReason::ToolUse,
        ))],
        vec![tool],
        None,
        1,
        5,
        CancellationToken::new(),
    );
    let id = harness.run_id;
    let done = run(harness).await;
    let outcome = done.outcome;
    assert_eq!(outcome.status, RunStatus::MaxIterations);
    assert_eq!(outcome.error_kind.as_deref(), Some("max_iterations"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        tool_json(&outcome.transcript, "c1")["content"],
        json!("once")
    );
    assert_eq!(done.llm.seen().len(), 1);
    let snap = done.supervisor.snapshot(id).unwrap();
    assert_eq!(snap.status, RunStatus::MaxIterations);
    assert_eq!(snap.iteration, 1);
    assert_eq!(snap.phase, Phase::React);
    assert_eq!(snap.error_kind.as_deref(), Some("max_iterations"));
}

#[tokio::test]
async fn cancel_before_the_tool_does_not_call_it() {
    let (tool, calls) = ScriptedTool::new("echo", vec![Ok("nope".into())]);
    let harness = harness(
        vec![Step::Cancel(assistant(
            None,
            vec![tool_call("c1", "echo", Some(json!({})))],
            StopReason::ToolUse,
        ))],
        vec![tool],
        None,
        4,
        5,
        CancellationToken::new(),
    );
    let outcome = run(harness).await.outcome;
    assert_eq!(outcome.status, RunStatus::Cancelled);
    assert!(outcome.error_kind.is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(outcome
        .transcript
        .iter()
        .all(|message| !matches!(message, Message::Tool { .. })));
}

#[tokio::test]
async fn a_hidden_registered_tool_is_not_called() {
    let (hidden, hidden_calls) = ScriptedTool::new("hidden", vec![Ok("secret".into())]);
    let (visible, visible_calls) = ScriptedTool::new("visible", vec![Ok("pub".into())]);
    let harness = harness(
        vec![
            Step::Ok(assistant(
                None,
                vec![tool_call("c1", "hidden", Some(json!({})))],
                StopReason::ToolUse,
            )),
            Step::Ok(assistant(Some("done"), vec![], StopReason::EndTurn)),
        ],
        vec![hidden, visible],
        Some(vec!["visible".into()]),
        4,
        5,
        CancellationToken::new(),
    );
    let done = run(harness).await;
    let outcome = done.outcome;
    assert_eq!(outcome.status, RunStatus::Succeeded);
    assert_eq!(hidden_calls.load(Ordering::SeqCst), 0);
    assert_eq!(visible_calls.load(Ordering::SeqCst), 0);
    assert_eq!(done.llm.seen()[0].tool_names, ["visible"]);
    let obs = tool_json(&outcome.transcript, "c1");
    assert_eq!(obs["ok"], json!(false));
    assert_eq!(obs["error"]["kind"], json!("not_found"));
    assert_eq!(obs["error"]["message"], json!("no such tool: hidden"));
}

#[tokio::test]
async fn finish_ok_stops_before_later_tool_calls() {
    let (finish, finish_calls) = ScriptedTool::new("finish", vec![Ok("the answer".into())]);
    let (echo, echo_calls) = ScriptedTool::new("echo", vec![Ok("later".into())]);
    let harness = harness(
        vec![Step::Ok(assistant(
            None,
            vec![
                tool_call("c1", "finish", Some(json!({"answer": "the answer"}))),
                tool_call("c2", "echo", Some(json!({}))),
            ],
            StopReason::ToolUse,
        ))],
        vec![finish, echo],
        None,
        4,
        5,
        CancellationToken::new(),
    );
    let done = run(harness).await;
    let outcome = done.outcome;
    assert_eq!(outcome.status, RunStatus::Succeeded);
    assert_eq!(outcome.summary, "the answer");
    assert_eq!(finish_calls.load(Ordering::SeqCst), 1);
    assert_eq!(echo_calls.load(Ordering::SeqCst), 0);
    assert_eq!(done.llm.seen().len(), 1);
    assert_eq!(tool_json(&outcome.transcript, "c1")["ok"], json!(true));
}

#[tokio::test]
async fn llm_errors_map_onto_the_stop_table() {
    let cases = [
        (LlmErrorKind::Auth, "auth"),
        (LlmErrorKind::BadRequest, "bad_request"),
        (LlmErrorKind::Unavailable, "unavailable"),
        (LlmErrorKind::Timeout, "timeout"),
    ];
    for (kind, expected) in cases {
        let harness = harness(
            vec![Step::Err(Error::Llm {
                kind,
                message: format!("HTTP {expected}"),
            })],
            Vec::new(),
            None,
            4,
            5,
            CancellationToken::new(),
        );
        let outcome = run(harness).await.outcome;
        assert_eq!(outcome.status, RunStatus::Failed, "{expected}");
        assert_eq!(outcome.error_kind.as_deref(), Some(expected));
    }
}

fn scratch() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "agentsmith-runtime-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn foreground_config(data: &Path, work: &Path, shell: Option<bool>) -> Config {
    let mut config = Config::default();
    config.harness.data_dir = data.to_path_buf();
    config.harness.workdir = work.to_path_buf();
    config.llm.model = "test-model".into();
    config.tools.shell.enabled = shell;
    config
}

#[tokio::test]
async fn foreground_run_writes_jsonl_and_shell_follows_the_unset_flag() {
    let dir = scratch();
    let work = dir.join("work");
    let data = dir.join("data");
    std::fs::create_dir_all(&work).unwrap();
    let llm = FakeLlm::new(vec![Step::Ok(assistant(
        Some("hello"),
        vec![],
        StopReason::EndTurn,
    ))]);
    let outcome = run_foreground(ForegroundParams {
        task: "do the thing".into(),
        spec: AgentSpec::react_default(),
        config: foreground_config(&data, &work, None),
        model: "test-model".into(),
        workdir: work.clone(),
        llm: llm.clone(),
        cancel: CancellationToken::new(),
    })
    .await
    .unwrap();
    assert_eq!(outcome.status, RunStatus::Succeeded);
    assert_eq!(outcome.summary, "hello");
    assert_eq!(
        llm.seen()[0].tool_names,
        ["finish", "list_dir", "read_file", "shell", "write_file"]
    );
    let runs = data.join("runs");
    let files: Vec<_> = std::fs::read_dir(&runs)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(files.len(), 1);
    let text = std::fs::read_to_string(&files[0]).unwrap();
    assert!(text.contains("\"role\":\"user\""));
    assert!(text.contains("do the thing"));
    assert!(text.contains("\"event\":\"finished\""));
    assert!(text.contains("\"status\":\"succeeded\""));
    assert!(text
        .lines()
        .last()
        .unwrap()
        .contains("\"event\":\"finished\""));

    let data_off = dir.join("data-off");
    let llm_off = FakeLlm::new(vec![Step::Ok(assistant(
        Some("off"),
        vec![],
        StopReason::EndTurn,
    ))]);
    let outcome = run_foreground(ForegroundParams {
        task: "do the thing".into(),
        spec: AgentSpec::react_default(),
        config: foreground_config(&data_off, &work, Some(false)),
        model: "test-model".into(),
        workdir: work,
        llm: llm_off.clone(),
        cancel: CancellationToken::new(),
    })
    .await
    .unwrap();
    assert_eq!(outcome.summary, "off");
    assert_eq!(
        llm_off.seen()[0].tool_names,
        ["finish", "list_dir", "read_file", "write_file"]
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn foreground_rejects_an_unavailable_spec_tool_before_writing_jsonl() {
    let dir = scratch();
    let work = dir.join("work");
    let data = dir.join("data");
    std::fs::create_dir_all(&work).unwrap();
    let mut spec = AgentSpec::react_default();
    spec.tools = Some(vec!["nope".into()]);
    let err = match run_foreground(ForegroundParams {
        task: "do the thing".into(),
        spec,
        config: foreground_config(&data, &work, Some(false)),
        model: "test-model".into(),
        workdir: work,
        llm: FakeLlm::new(vec![]),
        cancel: CancellationToken::new(),
    })
    .await
    {
        Err(err) => err,
        Ok(_) => panic!("expected an unavailable spec tool to fail the run"),
    };
    assert!(matches!(err, Error::Config(_)), "{err}");
    assert!(err.to_string().contains("nope"), "{err}");
    assert!(!data.join("runs").exists());
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn foreground_rejects_an_oversized_token_limit_before_writing_jsonl() {
    let dir = scratch();
    let data = dir.join("data");
    let mut config = foreground_config(&data, &dir, Some(false));
    config.llm.max_output_tokens = u64::from(u32::MAX) + 1;
    let err = match run_foreground(ForegroundParams {
        task: "do the thing".into(),
        spec: AgentSpec::react_default(),
        config,
        model: "test-model".into(),
        workdir: dir.clone(),
        llm: FakeLlm::new(vec![]),
        cancel: CancellationToken::new(),
    })
    .await
    {
        Err(err) => err,
        Ok(_) => panic!("expected an oversized token limit to fail the run"),
    };
    assert!(err.to_string().contains("max_output_tokens"), "{err}");
    assert!(!data.join("runs").exists());
    let _ = std::fs::remove_dir_all(dir);
}
