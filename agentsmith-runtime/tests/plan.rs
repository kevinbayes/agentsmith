//! Plan walker, join-all, and `spawn_subagents`. No network.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use agentsmith_core::config::{Config, FailurePolicy};
use agentsmith_core::{Error, LlmErrorKind, ToolErrorKind};
use agentsmith_llm::{
    Completion, CompletionRequest, Llm, Message, StopReason, ToolCall, ToolChoice,
};
use agentsmith_runtime::{
    run_foreground, AgentSpec, CancellationToken, ChildOrigin, ChildSpec, ForegroundParams,
    LoopLimits, LoopMode, PlanExecuteLoop, ReactLoop, RunContext, RunEvents, RunId, RunKind,
    RunOutcome, RunSnapshot, RunStatus, RunTools, SpawnRequest, SpawnerConfig, SubagentSpawner,
    Supervisor, ToolFilter, Transcript,
};
use agentsmith_tools::{Tool, ToolCtx, ToolDefinition, ToolOutput, ToolRegistry};
use serde_json::{json, Value};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}

#[derive(Clone, Debug)]
enum Choice {
    Auto,
    Function(String),
}

struct Seen {
    choice: Choice,
    tools: Vec<String>,
    messages: Vec<Message>,
}

enum Queued {
    Completion(Completion),
    Err(Error),
    CancelWalker(Completion),
}

struct QueueLlm {
    steps: Mutex<VecDeque<Queued>>,
    seen: Mutex<Vec<Seen>>,
    walker: CancellationToken,
}

impl QueueLlm {
    fn new(steps: Vec<Queued>, walker: CancellationToken) -> Arc<Self> {
        Arc::new(Self {
            steps: Mutex::new(steps.into()),
            seen: Mutex::new(Vec::new()),
            walker,
        })
    }

    fn seen(&self) -> Vec<Seen> {
        lock(&self.seen).clone()
    }
}

impl Clone for Seen {
    fn clone(&self) -> Self {
        Self {
            choice: self.choice.clone(),
            tools: self.tools.clone(),
            messages: self.messages.clone(),
        }
    }
}

#[async_trait::async_trait]
impl Llm for QueueLlm {
    async fn complete(&self, request: CompletionRequest) -> Result<Completion, Error> {
        let choice = match request.tool_choice {
            ToolChoice::Auto => Choice::Auto,
            ToolChoice::Function { name } => Choice::Function(name),
        };
        lock(&self.seen).push(Seen {
            choice,
            tools: request.tools.into_iter().map(|tool| tool.name).collect(),
            messages: request.messages,
        });
        let step = lock(&self.steps)
            .pop_front()
            .expect("complete called more times than scripted");
        match step {
            Queued::Completion(completion) => Ok(completion),
            Queued::Err(err) => Err(err),
            Queued::CancelWalker(completion) => {
                // The walker token, not this request's child token.
                self.walker.cancel();
                Ok(completion)
            }
        }
    }
}

struct SleepyLlm {
    delay: Duration,
}

#[async_trait::async_trait]
impl Llm for SleepyLlm {
    async fn complete(&self, _request: CompletionRequest) -> Result<Completion, Error> {
        tokio::time::sleep(self.delay).await;
        Ok(end_completion("late"))
    }
}

struct Probe;

#[async_trait::async_trait]
impl Tool for Probe {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "probe".into(),
            description: "probe".into(),
            parameters: json!({"type": "object", "additionalProperties": false}),
        }
    }

    async fn call(&self, _ctx: ToolCtx, _arguments: Value) -> Result<ToolOutput, Error> {
        Ok(ToolOutput {
            content: "pong".into(),
        })
    }
}

struct NamedTool {
    name: String,
}

#[async_trait::async_trait]
impl Tool for NamedTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.clone(),
            description: self.name.clone(),
            parameters: json!({"type": "object", "additionalProperties": false}),
        }
    }

    async fn call(&self, _ctx: ToolCtx, _arguments: Value) -> Result<ToolOutput, Error> {
        Ok(ToolOutput {
            content: "ok".into(),
        })
    }
}

fn base_limits() -> LoopLimits {
    LoopLimits {
        react_max_iterations: 25,
        step_max_iterations: 4,
        planner_attempts: 2,
        max_consecutive_tool_errors: 5,
        max_subagent_depth: 2,
        max_fanout: 8,
        max_live_subagents: 8,
        spawn_timeout: Duration::from_secs(30),
        failure_policy: FailurePolicy::Continue,
    }
}

fn spec() -> AgentSpec {
    AgentSpec {
        id: "planner".into(),
        name: "Planner".into(),
        description: String::new(),
        mode: LoopMode::Plan,
        system_prompt: "Be precise.".into(),
        tools: None,
        subagent_tools: None,
        model: None,
    }
}

struct World {
    dir: PathBuf,
    llm: Arc<QueueLlm>,
    supervisor: Arc<Supervisor>,
    spawner: Arc<SubagentSpawner>,
    registry: ToolRegistry,
    limits: LoopLimits,
    spec: AgentSpec,
    run_id: RunId,
    cancel: CancellationToken,
}

fn harness(spec: AgentSpec, limits: LoopLimits, steps: Vec<Queued>) -> World {
    let dir = scratch();
    let cancel = CancellationToken::new();
    let llm = QueueLlm::new(steps, cancel.clone());
    let supervisor = Arc::new(Supervisor::new());
    let registry = ToolRegistry::new();
    let spawner = SubagentSpawner::new(SpawnerConfig {
        registry: registry.clone(),
        llm: llm.clone(),
        model: "m".into(),
        max_output_tokens: 64,
        temperature: None,
        max_output_bytes: 32 * 1024,
        workdir: dir.clone(),
        data_dir: dir.join("data"),
        spec: spec.clone(),
        allow: None,
        limits: limits.clone(),
        events: supervisor.clone() as Arc<dyn RunEvents>,
    });
    spawner.install().unwrap();
    World {
        dir,
        llm,
        supervisor,
        spawner,
        registry,
        limits,
        spec,
        run_id: agentsmith_core::id::new_id(),
        cancel,
    }
}

fn scratch() -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "agentsmith-plan-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

async fn run_plan(world: &World) -> RunOutcome {
    let ctx = RunContext {
        run_id: world.run_id,
        parent_run_id: None,
        depth: 0,
        name: world.spec.name.clone(),
        spec: world.spec.clone(),
        task: "ship it".into(),
        extra_context: None,
        model: "m".into(),
        max_output_tokens: 64,
        temperature: None,
        max_output_bytes: 32 * 1024,
        max_iterations: world.limits.react_max_iterations,
        workdir: world.dir.clone(),
        llm: world.llm.clone(),
        tools: RunTools::resolve(world.registry.clone(), None, None).unwrap(),
        limits: world.limits.clone(),
        cancel: world.cancel.clone(),
        events: world.supervisor.clone() as Arc<dyn RunEvents>,
        transcript: Transcript::empty(),
        spawner: Some(world.spawner.clone()),
    };
    PlanExecuteLoop.run(ctx).await.expect("plan run")
}

fn children(world: &World) -> Vec<RunSnapshot> {
    world
        .supervisor
        .snapshots()
        .into_iter()
        .filter(|snap| snap.run_id != world.run_id)
        .collect()
}

fn walker(world: &World) -> RunSnapshot {
    world
        .supervisor
        .snapshot(world.run_id)
        .expect("walker snapshot")
}

fn plan(steps: Vec<Value>) -> Queued {
    Queued::Completion(Completion {
        message: Message::Assistant {
            content: None,
            tool_calls: vec![call(
                "plan",
                "submit_plan",
                json!({"goal": "ship", "steps": steps}),
            )],
        },
        stop: StopReason::ToolUse,
    })
}

fn end(text: &str) -> Queued {
    Queued::Completion(end_completion(text))
}

fn end_completion(text: &str) -> Completion {
    Completion {
        message: Message::Assistant {
            content: Some(text.to_string()),
            tool_calls: Vec::new(),
        },
        stop: StopReason::EndTurn,
    }
}

fn llm_err(message: &str) -> Queued {
    Queued::Err(Error::Llm {
        kind: LlmErrorKind::Unavailable,
        message: message.into(),
    })
}

fn call(id: &str, name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments_raw: serde_json::to_string(&arguments).unwrap(),
        arguments: Some(arguments),
    }
}

fn step(id: &str, instruction: &str) -> Value {
    json!({"id": id, "title": id, "instruction": instruction})
}

fn step_dep(id: &str, instruction: &str, deps: &[&str]) -> Value {
    json!({"id": id, "title": id, "instruction": instruction, "depends_on": deps})
}

fn step_group(id: &str, instruction: &str, group: &str) -> Value {
    json!({"id": id, "title": id, "instruction": instruction, "parallel_group": group})
}

fn users(seen: &Seen) -> String {
    seen.messages
        .iter()
        .filter_map(|message| match message {
            Message::User { content } => Some(content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn systems(seen: &Seen) -> String {
    seen.messages
        .iter()
        .filter_map(|message| match message {
            Message::System { content } => Some(content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn cleanup(dir: PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

const REPAIR: &str = "The previous response was not a valid submit_plan call. Call submit_plan with a JSON object matching the schema. Do not explain.";

#[tokio::test]
async fn group_member_depth_matches_the_walker_even_at_max_zero() {
    let mut limits = base_limits();
    limits.max_subagent_depth = 0;
    let world = harness(
        spec(),
        limits,
        vec![
            plan(vec![
                step_group("a", "do a", "g"),
                step_group("b", "do b", "g"),
            ]),
            end("done"),
            end("done"),
        ],
    );
    let outcome = run_plan(&world).await;
    assert_eq!(outcome.status, RunStatus::Succeeded, "{}", outcome.summary);
    let seen = world.llm.seen();
    assert!(matches!(seen[0].choice, Choice::Function(ref name) if name == "submit_plan"));
    assert_eq!(seen[0].tools, ["submit_plan"]);
    let kids = children(&world);
    assert_eq!(kids.len(), 2, "{kids:?}");
    for kid in &kids {
        assert_eq!(kid.parent_run_id, Some(world.run_id));
        assert_eq!(kid.depth, 0, "walker passes its depth and does not add 1");
        assert_eq!(kid.status, RunStatus::Succeeded);
    }
    assert!(seen
        .iter()
        .skip(1)
        .all(|item| !systems(item).contains("You are subagent")));
    let snap = walker(&world);
    assert_eq!(snap.iteration, 0);
    assert_eq!(snap.max_iterations, 4);
    assert_eq!(snap.tool_use.get("spawn_subagents").copied(), Some(1));
    assert_eq!(world.spawner.live(), 0);
    cleanup(world.dir);
}

#[tokio::test]
async fn succeeded_chain_reports_prior_results_and_counts_joins() {
    let world = harness(
        spec(),
        base_limits(),
        vec![
            plan(vec![step("a", "do-a"), step_dep("b", "do-b", &["a"])]),
            end("alpha-summary"),
            end("beta-summary"),
        ],
    );
    let outcome = run_plan(&world).await;
    assert_eq!(outcome.status, RunStatus::Succeeded, "{}", outcome.summary);
    let seen = world.llm.seen();
    let prior = seen
        .iter()
        .find(|item| users(item).contains("do-b"))
        .expect("step b");
    let text = users(prior);
    assert!(text.contains("Prior results"), "{text}");
    assert!(text.contains("- a: succeeded"), "{text}");
    assert!(text.contains("alpha-summary"), "{text}");
    assert!(!text.contains("tool_call"), "{text}");
    let snap = walker(&world);
    assert_eq!(snap.iteration, 0);
    assert_eq!(snap.max_iterations, 4);
    assert_ne!(snap.max_iterations, world.limits.react_max_iterations);
    assert_eq!(snap.tool_use.get("spawn_subagents").copied(), Some(2));
    for kid in children(&world) {
        assert_eq!(kid.max_iterations, 4);
        assert_eq!(kid.depth, 0);
    }
    let runs = world.dir.join("data").join("runs");
    let logs: Vec<String> = std::fs::read_dir(&runs)
        .unwrap()
        .map(|entry| std::fs::read_to_string(entry.unwrap().path()).unwrap())
        .collect();
    assert!(logs.iter().any(|log| log.contains("do-a")), "{logs:?}");
    assert!(logs.iter().any(|log| log.contains("do-b")), "{logs:?}");
    cleanup(world.dir);
}

#[tokio::test]
async fn one_success_and_a_skip_is_partial_and_the_dependent_has_no_run() {
    let world = harness(
        spec(),
        base_limits(),
        vec![
            plan(vec![
                step("a", "do-a"),
                step("b", "do-b"),
                step_dep("c", "do-c", &["b"]),
            ]),
            end("alpha"),
            llm_err("step b broke"),
        ],
    );
    let outcome = run_plan(&world).await;
    assert_eq!(outcome.status, RunStatus::Partial, "{}", outcome.summary);
    let kids = children(&world);
    assert!(kids
        .iter()
        .any(|kid| kid.name == "a" && kid.status == RunStatus::Succeeded));
    assert!(kids
        .iter()
        .any(|kid| kid.name == "b" && kid.status == RunStatus::Failed));
    assert!(kids.iter().all(|kid| kid.name != "c"));
    cleanup(world.dir);
}

#[tokio::test]
async fn a_failed_dependency_skips_dependents_and_fails_the_root() {
    let world = harness(
        spec(),
        base_limits(),
        vec![
            plan(vec![step("a", "do-a"), step_dep("b", "do-b", &["a"])]),
            llm_err("nope"),
        ],
    );
    let outcome = run_plan(&world).await;
    assert_eq!(outcome.status, RunStatus::Failed, "{}", outcome.summary);
    let kids = children(&world);
    assert_eq!(kids.len(), 1);
    assert_eq!(kids[0].name, "a");
    assert_eq!(kids[0].status, RunStatus::Failed);
    cleanup(world.dir);
}

#[tokio::test]
async fn root_cancel_does_not_aggregate_and_skips_creating_the_next_run() {
    let cancel = CancellationToken::new();
    let dir = scratch();
    let supervisor = Arc::new(Supervisor::new());
    let llm = QueueLlm::new(
        vec![
            plan(vec![step("a", "do-a"), step("b", "do-b")]),
            Queued::CancelWalker(end_completion("stopping")),
        ],
        cancel.clone(),
    );
    let registry = ToolRegistry::new();
    let limits = base_limits();
    let spec = spec();
    let spawner = SubagentSpawner::new(SpawnerConfig {
        registry: registry.clone(),
        llm: llm.clone(),
        model: "m".into(),
        max_output_tokens: 64,
        temperature: None,
        max_output_bytes: 32 * 1024,
        workdir: dir.clone(),
        data_dir: dir.join("data"),
        spec: spec.clone(),
        allow: None,
        limits: limits.clone(),
        events: supervisor.clone() as Arc<dyn RunEvents>,
    });
    spawner.install().unwrap();
    let run_id = agentsmith_core::id::new_id();
    let outcome = PlanExecuteLoop
        .run(RunContext {
            run_id,
            parent_run_id: None,
            depth: 0,
            name: spec.name.clone(),
            spec,
            task: "ship it".into(),
            extra_context: None,
            model: "m".into(),
            max_output_tokens: 64,
            temperature: None,
            max_output_bytes: 32 * 1024,
            max_iterations: limits.react_max_iterations,
            workdir: dir.clone(),
            llm,
            tools: RunTools::resolve(registry, None, None).unwrap(),
            limits,
            cancel,
            events: supervisor.clone() as Arc<dyn RunEvents>,
            transcript: Transcript::empty(),
            spawner: Some(spawner),
        })
        .await
        .unwrap();
    assert_eq!(outcome.status, RunStatus::Cancelled);
    assert!(outcome.error_kind.is_none());
    assert_eq!(outcome.summary, "cancelled");
    let snaps = supervisor.snapshots();
    assert_eq!(snaps.len(), 2, "{snaps:?}");
    assert!(snaps
        .iter()
        .any(|snap| snap.run_id == run_id && snap.status == RunStatus::Cancelled));
    assert!(snaps
        .iter()
        .any(|snap| snap.name == "a" && snap.status == RunStatus::Cancelled));
    assert!(snaps.iter().all(|snap| snap.name != "b"));
    cleanup(dir);
}

#[tokio::test]
async fn oversized_group_is_plan_invalid_and_starts_nothing() {
    let mut limits = base_limits();
    limits.planner_attempts = 1;
    limits.max_fanout = 2;
    let steps = (0..3)
        .map(|index| step_group(&format!("s{index}"), "work", "g"))
        .collect();
    let world = harness(spec(), limits, vec![plan(steps)]);
    let outcome = run_plan(&world).await;
    assert_eq!(outcome.status, RunStatus::Failed);
    assert_eq!(outcome.error_kind.as_deref(), Some("plan_invalid"));
    assert!(
        outcome.summary.contains("max_fanout"),
        "{}",
        outcome.summary
    );
    assert!(children(&world).is_empty());
    assert_eq!(world.llm.seen().len(), 1);
    assert_eq!(world.spawner.live(), 0);
    cleanup(world.dir);
}

#[tokio::test]
async fn depends_on_inside_a_parallel_group_is_plan_invalid() {
    let mut limits = base_limits();
    limits.planner_attempts = 1;
    let world = harness(
        spec(),
        limits,
        vec![plan(vec![
            step_group("a", "do a", "g"),
            json!({
                "id": "b",
                "title": "b",
                "instruction": "do b",
                "parallel_group": "g",
                "depends_on": ["a"]
            }),
        ])],
    );
    let outcome = run_plan(&world).await;
    assert_eq!(outcome.status, RunStatus::Failed);
    assert_eq!(outcome.error_kind.as_deref(), Some("plan_invalid"));
    assert!(
        outcome.summary.contains("parallel_group"),
        "{}",
        outcome.summary
    );
    assert!(children(&world).is_empty());
    assert_eq!(world.spawner.live(), 0);
    cleanup(world.dir);
}

#[tokio::test]
async fn planner_repair_then_plan_invalid_on_the_second_failure() {
    let world = harness(
        spec(),
        base_limits(),
        vec![
            end("I will explain instead"),
            plan(vec![step("a", "one"), step("a", "two")]),
        ],
    );
    let outcome = run_plan(&world).await;
    assert_eq!(outcome.status, RunStatus::Failed);
    assert_eq!(outcome.error_kind.as_deref(), Some("plan_invalid"));
    let seen = world.llm.seen();
    assert_eq!(seen.len(), 2);
    assert!(users(&seen[1]).contains(REPAIR), "{}", users(&seen[1]));
    assert!(children(&world).is_empty());
    cleanup(world.dir);
}

#[tokio::test]
async fn a_step_uses_step_max_iterations_not_the_react_cap() {
    let mut limits = base_limits();
    limits.step_max_iterations = 1;
    limits.react_max_iterations = 25;
    let mut spec = spec();
    spec.tools = Some(vec!["probe".into()]);
    let world = harness(
        spec,
        limits,
        vec![
            plan(vec![step("a", "probe it")]),
            Queued::Completion(Completion {
                message: Message::Assistant {
                    content: None,
                    tool_calls: vec![call("t", "probe", json!({}))],
                },
                stop: StopReason::ToolUse,
            }),
        ],
    );
    world.registry.register(Arc::new(Probe)).unwrap();
    let outcome = run_plan(&world).await;
    assert_eq!(
        world.llm.seen().len(),
        2,
        "the react cap would ask for another turn"
    );
    let kids = children(&world);
    assert_eq!(kids.len(), 1);
    assert_eq!(kids[0].status, RunStatus::MaxIterations);
    assert_eq!(kids[0].iteration, 1);
    assert_eq!(kids[0].max_iterations, 1);
    assert_eq!(kids[0].tool_use.get("probe").copied(), Some(1));
    assert_eq!(walker(&world).tool_use.get("probe").copied(), None);
    assert_eq!(walker(&world).iteration, 0);
    assert_eq!(walker(&world).max_iterations, 1);
    assert_eq!(outcome.status, RunStatus::Failed);
    cleanup(world.dir);
}

fn spawn_turn(agents: Value) -> Queued {
    Queued::Completion(Completion {
        message: Message::Assistant {
            content: None,
            tool_calls: vec![call("s", "spawn_subagents", json!({"agents": agents}))],
        },
        stop: StopReason::ToolUse,
    })
}

async fn run_spawn_parent(world: &World, depth: u32, max_iters: u32) -> RunOutcome {
    let tools = RunTools::resolve_for(
        world.registry.clone(),
        ToolFilter {
            allow: None,
            spec_tools: world.spec.tools.as_deref(),
            subagent_tools: None,
            kind: RunKind::Step,
            depth,
            max_subagent_depth: world.limits.max_subagent_depth,
        },
    )
    .unwrap();
    let run_id = agentsmith_core::id::new_id();
    ReactLoop
        .run(RunContext {
            run_id,
            parent_run_id: None,
            depth,
            name: "step".into(),
            spec: world.spec.clone(),
            task: "delegate".into(),
            extra_context: None,
            model: "m".into(),
            max_output_tokens: 64,
            temperature: None,
            max_output_bytes: 32 * 1024,
            max_iterations: max_iters,
            workdir: world.dir.clone(),
            llm: world.llm.clone(),
            tools,
            limits: world.limits.clone(),
            cancel: world.cancel.clone(),
            events: world.supervisor.clone() as Arc<dyn RunEvents>,
            transcript: Transcript::empty(),
            spawner: Some(world.spawner.clone()),
        })
        .await
        .unwrap()
}

fn tool_body(outcome: &RunOutcome) -> Value {
    outcome
        .transcript
        .iter()
        .find_map(|message| match message {
            Message::Tool { content, .. } => Some(serde_json::from_str(content).unwrap()),
            _ => None,
        })
        .expect("tool message")
}

#[tokio::test]
async fn tool_path_assigns_caller_depth_plus_one() {
    let world = harness(
        spec(),
        base_limits(),
        vec![
            spawn_turn(json!([{"name": "reader", "task": "read it", "context": "page 1"}])),
            end("child-text"),
            end("parent-done"),
        ],
    );
    world
        .registry
        .register(Arc::new(NamedTool {
            name: "read_file".into(),
        }))
        .unwrap();
    let outcome = run_spawn_parent(&world, 0, 4).await;
    assert_eq!(outcome.status, RunStatus::Succeeded);
    let kids = world.supervisor.snapshots();
    let child = kids.iter().find(|snap| snap.name == "reader").unwrap();
    assert_eq!(child.depth, 1);
    let seen = world.llm.seen();
    let child_seen = seen
        .iter()
        .find(|item| systems(item).contains("You are subagent reader at depth 1"))
        .expect("child prompt");
    assert!(systems(child_seen).contains("Do not assume you can see the parent transcript."));
    assert!(users(child_seen).contains("read it"));
    assert!(users(child_seen).contains("page 1"));
    assert!(child_seen.tools.iter().any(|name| name == "read_file"));
    assert!(child_seen
        .tools
        .iter()
        .any(|name| name == "spawn_subagents"));
    assert_eq!(world.spawner.live(), 0);
    cleanup(world.dir);
}

#[tokio::test]
async fn depth_past_max_starts_zero_children_and_returns_limit() {
    let world = harness(spec(), base_limits(), Vec::new());
    let run_id = agentsmith_core::id::new_id();
    let _guard = world.spawner.enter(run_id, 2, "sys".into());
    let before = world.spawner.live();
    let tool = world.registry.get("spawn_subagents").unwrap();
    let err = tool
        .call(
            ToolCtx {
                run_id,
                workdir: world.dir.clone(),
                cancel: CancellationToken::new(),
            },
            json!({"agents": [{"name": "a", "task": "t"}]}),
        )
        .await
        .unwrap_err();
    match err {
        Error::Tool { kind, .. } => assert_eq!(kind, ToolErrorKind::Limit),
        other => panic!("{other}"),
    }
    assert_eq!(world.spawner.live(), before);
    assert!(world.supervisor.snapshots().is_empty());
    cleanup(world.dir);
}

#[tokio::test]
async fn fanout_and_live_caps_start_zero_children() {
    let mut limits = base_limits();
    limits.max_fanout = 1;
    limits.max_live_subagents = 1;
    let world = harness(spec(), limits, Vec::new());
    let run_id = agentsmith_core::id::new_id();
    let _guard = world.spawner.enter(run_id, 0, "sys".into());
    let tool = world.registry.get("spawn_subagents").unwrap();
    let err = tool
        .call(
            ToolCtx {
                run_id,
                workdir: world.dir.clone(),
                cancel: CancellationToken::new(),
            },
            json!({"agents": [{"name": "a", "task": "t"}, {"name": "b", "task": "u"}]}),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::Tool {
                kind: ToolErrorKind::Limit,
                ..
            }
        ),
        "{err}"
    );
    assert_eq!(world.spawner.live(), 0);
    assert!(world.supervisor.snapshots().is_empty());
    cleanup(world.dir);

    let mut limits = base_limits();
    limits.max_live_subagents = 1;
    let world = harness(spec(), limits, Vec::new());
    let run_id = agentsmith_core::id::new_id();
    let _guard = world.spawner.enter(run_id, 0, "sys".into());
    let tool = world.registry.get("spawn_subagents").unwrap();
    let err = tool
        .call(
            ToolCtx {
                run_id,
                workdir: world.dir.clone(),
                cancel: CancellationToken::new(),
            },
            json!({"agents": [{"name": "a", "task": "t"}, {"name": "b", "task": "u"}]}),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::Tool {
                kind: ToolErrorKind::Limit,
                ..
            }
        ),
        "{err}"
    );
    assert_eq!(world.spawner.live(), 0);
    assert!(world.supervisor.snapshots().is_empty());
    cleanup(world.dir);
}

#[tokio::test]
async fn spawn_is_absent_when_the_child_is_at_max_depth() {
    let world = harness(
        spec(),
        base_limits(),
        vec![
            spawn_turn(json!([{"name": "reader", "task": "read it"}])),
            end("child-text"),
            end("parent-done"),
        ],
    );
    world
        .registry
        .register(Arc::new(NamedTool {
            name: "read_file".into(),
        }))
        .unwrap();
    run_spawn_parent(&world, 1, 4).await;
    let seen = world.llm.seen();
    let child_seen = seen
        .iter()
        .find(|item| systems(item).contains("You are subagent reader at depth 2"))
        .expect("child at depth 2");
    assert!(child_seen
        .tools
        .iter()
        .all(|name| name != "spawn_subagents"));
    assert!(child_seen.tools.iter().any(|name| name == "read_file"));
    let child = world
        .supervisor
        .snapshots()
        .into_iter()
        .find(|snap| snap.name == "reader")
        .unwrap();
    assert_eq!(child.depth, 2);
    cleanup(world.dir);
}

#[tokio::test]
async fn subagent_tools_cannot_grant_and_foreground_rejects_it_at_start() {
    let mut spec = spec();
    spec.tools = Some(vec!["read_file".into()]);
    spec.subagent_tools = Some(vec!["shell".into()]);
    let world = harness(spec, base_limits(), Vec::new());
    world
        .registry
        .register(Arc::new(NamedTool {
            name: "read_file".into(),
        }))
        .unwrap();
    let err = match world
        .spawner
        .spawn_and_join(SpawnRequest {
            parent: world.run_id,
            child_depth: 1,
            agents: vec![ChildSpec {
                name: "reader".into(),
                task: "read".into(),
                context: None,
            }],
            timeout: Duration::from_secs(5),
            failure_policy: FailurePolicy::Continue,
            cancel: CancellationToken::new(),
            origin: ChildOrigin::Spawned,
        })
        .await
    {
        Err(err) => err,
        Ok(_) => panic!("expected subagent_tools to refuse a grant"),
    };
    assert!(matches!(err, Error::Config(_)), "{err}");
    assert!(err.to_string().contains("shell"), "{err}");
    assert_eq!(world.spawner.live(), 0);
    assert!(world.supervisor.snapshots().is_empty());

    let dir = scratch();
    let mut spec = AgentSpec::react_default();
    spec.tools = Some(vec!["read_file".into()]);
    spec.subagent_tools = Some(vec!["shell".into()]);
    let err = match run_foreground(ForegroundParams {
        task: "ship".into(),
        spec,
        config: {
            let mut config = Config::default();
            config.harness.data_dir = dir.join("data");
            config.harness.workdir = dir.clone();
            config.tools.shell.enabled = Some(false);
            config
        },
        model: "m".into(),
        workdir: dir.clone(),
        llm: QueueLlm::new(Vec::new(), CancellationToken::new()),
        cancel: CancellationToken::new(),
        mode: LoopMode::Plan,
    })
    .await
    {
        Err(err) => err,
        Ok(_) => panic!("expected foreground to reject a subagent_tools grant"),
    };
    assert!(err.to_string().contains("shell"), "{err}");
    assert!(!dir.join("data").join("runs").exists());
    cleanup(world.dir);
    cleanup(dir);
}

#[tokio::test]
async fn join_ok_is_true_only_when_every_child_succeeded() {
    let world = harness(
        spec(),
        base_limits(),
        vec![
            spawn_turn(json!([
                {"name": "one", "task": "a"},
                {"name": "two", "task": "b"}
            ])),
            end("alpha"),
            end("beta"),
        ],
    );
    let outcome = run_spawn_parent(&world, 0, 1).await;
    let body = tool_body(&outcome);
    assert_eq!(body["ok"], true);
    assert_eq!(body["joined"], "all");
    assert_eq!(body["agents"].as_array().unwrap().len(), 2);
    cleanup(world.dir);

    let world = harness(
        spec(),
        base_limits(),
        vec![
            spawn_turn(json!([
                {"name": "one", "task": "a"},
                {"name": "two", "task": "b"}
            ])),
            end("alpha"),
            llm_err("sk-testsecretvalue leaked"),
        ],
    );
    let outcome = run_spawn_parent(&world, 0, 1).await;
    let raw = outcome
        .transcript
        .iter()
        .find_map(|message| match message {
            Message::Tool { content, .. } => Some(content.clone()),
            _ => None,
        })
        .unwrap();
    assert!(!raw.contains("sk-testsecretvalue"), "{raw}");
    assert!(raw.contains("[redacted]"), "{raw}");
    let body: Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(body["ok"], false);
    assert_eq!(body["agents"].as_array().unwrap().len(), 2);
    assert_eq!(world.spawner.live(), 0);
    cleanup(world.dir);
}

#[tokio::test]
async fn a_child_that_ignores_cancel_releases_its_slot_after_grace() {
    let dir = scratch();
    let supervisor = Arc::new(Supervisor::new());
    let llm = Arc::new(SleepyLlm {
        delay: Duration::from_secs(5),
    });
    let limits = base_limits();
    let spawner = SubagentSpawner::new(SpawnerConfig {
        registry: ToolRegistry::new(),
        llm,
        model: "m".into(),
        max_output_tokens: 32,
        temperature: None,
        max_output_bytes: 1024,
        workdir: dir.clone(),
        data_dir: dir.join("data"),
        spec: spec(),
        allow: None,
        limits: limits.clone(),
        events: supervisor.clone() as Arc<dyn RunEvents>,
    });
    spawner.install().unwrap();
    let before = spawner.live();
    let started = tokio::time::Instant::now();
    let joined = spawner
        .spawn_and_join(SpawnRequest {
            parent: agentsmith_core::id::new_id(),
            child_depth: 0,
            agents: vec![ChildSpec {
                name: "sleeper".into(),
                task: "ignore cancel".into(),
                context: None,
            }],
            timeout: Duration::from_millis(100),
            failure_policy: FailurePolicy::Continue,
            cancel: CancellationToken::new(),
            origin: ChildOrigin::Step,
        })
        .await
        .unwrap();
    let elapsed = started.elapsed();
    assert!(elapsed >= Duration::from_millis(1900), "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(4), "{elapsed:?}");
    assert_eq!(joined.agents.len(), 1);
    assert_eq!(joined.agents[0].status, RunStatus::TimedOut);
    assert_eq!(spawner.live(), before);
    let run_id = joined.agents[0].run_id;
    assert_eq!(
        supervisor.snapshot(run_id).unwrap().status,
        RunStatus::TimedOut
    );
    tokio::time::sleep(Duration::from_secs(4)).await;
    assert_eq!(spawner.live(), before);
    assert_eq!(
        supervisor.snapshot(run_id).unwrap().status,
        RunStatus::TimedOut
    );
    cleanup(dir);
}
