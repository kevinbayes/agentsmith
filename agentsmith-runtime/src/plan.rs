//! Deterministic plan walker. Not a ReAct loop: `iteration` stays 0.
//!
//! Solo steps and group members are child runs at the walker's depth. One join
//! is outstanding at a time.

use std::collections::BTreeMap;
use std::sync::Arc;

use agentsmith_core::Error;
use agentsmith_llm::{CompletionRequest, Message, ToolChoice, ToolDefinition};
use serde_json::{json, Value};

use crate::loop_::{
    poll_sessions, Phase, RunContext, RunEvents, RunId, RunMeta, RunOutcome, RunStatus,
};
use crate::run_tools::SPAWN_SUBAGENTS;
use crate::spawn::{cut_bytes, ChildOrigin, ChildSpec, SpawnRequest, SubagentSpawner};
use crate::CancellationToken;

const REPAIR: &str = "The previous response was not a valid submit_plan call. Call submit_plan with a JSON object matching the schema. Do not explain.";
const PLAN_ERROR_MAX: usize = 1024;
const SUMMARY_MAX: usize = 4 * 1024;
const MAX_STEPS: usize = 32;

#[derive(Debug, Default)]
pub struct PlanExecuteLoop;

impl PlanExecuteLoop {
    pub async fn run(&self, ctx: RunContext) -> Result<RunOutcome, Error> {
        execute(ctx).await
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepPhase {
    Pending,
    Running,
    Skipped,
    Done(RunStatus),
}

struct StepExec {
    id: String,
    instruction: String,
    depends_on: Vec<String>,
    parallel_group: Option<String>,
    status: StepPhase,
    run_id: Option<RunId>,
    summary: String,
}

struct Plan {
    goal: String,
    steps: Vec<StepExec>,
}

async fn execute(mut ctx: RunContext) -> Result<RunOutcome, Error> {
    let events = Arc::clone(&ctx.events);
    ctx.transcript
        .arm_unwind(Arc::clone(&events), ctx.run_id, Phase::Plan);
    let cancel = ctx.cancel.clone();
    let spawner = ctx
        .spawner
        .clone()
        .ok_or_else(|| Error::Loop("plan run has no spawner".into()))?;

    events
        .started(
            ctx.run_id,
            RunMeta {
                parent_run_id: ctx.parent_run_id,
                depth: ctx.depth,
                name: ctx.name.clone(),
                max_iterations: ctx.limits.step_max_iterations,
                phase: Phase::Plan,
            },
        )
        .await;
    events.phase(ctx.run_id, Phase::Plan).await;
    events.iteration(ctx.run_id, 0).await;

    ctx.transcript.push(Message::System {
        content: ctx.spec.system_prompt.clone(),
    })?;
    ctx.transcript.push(Message::User {
        content: ctx.task.clone(),
    })?;

    let plan = match plan_phase(&mut ctx, &events, &cancel).await? {
        PlanPhase::Ready(plan) => plan,
        PlanPhase::Stopped(outcome) => return Ok(outcome),
    };

    walk(&mut ctx, &events, &cancel, &spawner, plan).await
}

enum PlanPhase {
    Ready(Plan),
    Stopped(RunOutcome),
}

async fn plan_phase(
    ctx: &mut RunContext,
    events: &Arc<dyn RunEvents>,
    cancel: &CancellationToken,
) -> Result<PlanPhase, Error> {
    let llm = Arc::clone(&ctx.llm);
    let attempts = u32::from(ctx.limits.planner_attempts);
    let mut attempt = 0u32;
    let mut last_error = "submit_plan was not called".to_string();
    let definition = submit_plan_definition();

    loop {
        if cancel.is_cancelled() {
            return Ok(PlanPhase::Stopped(
                stop(ctx, events, RunStatus::Cancelled, "cancelled", None).await?,
            ));
        }
        if attempt >= attempts {
            return Ok(PlanPhase::Stopped(
                stop(
                    ctx,
                    events,
                    RunStatus::Failed,
                    cut_bytes(&last_error, PLAN_ERROR_MAX),
                    Some("plan_invalid"),
                )
                .await?,
            ));
        }
        attempt += 1;

        let request = CompletionRequest {
            model: ctx.model.clone(),
            messages: ctx.transcript.messages().to_vec(),
            tools: vec![definition.clone()],
            tool_choice: ToolChoice::Function {
                name: "submit_plan".to_string(),
            },
            max_output_tokens: ctx.max_output_tokens,
            temperature: ctx.temperature,
            cancel: cancel.clone(),
        };
        let completion = match llm.complete(request).await {
            Ok(completion) => completion,
            Err(err) => {
                if cancel.is_cancelled() || matches!(err, Error::Cancelled) {
                    return Ok(PlanPhase::Stopped(
                        stop(ctx, events, RunStatus::Cancelled, "cancelled", None).await?,
                    ));
                }
                if let Error::Llm { kind, message } = err {
                    return Ok(PlanPhase::Stopped(
                        stop(
                            ctx,
                            events,
                            RunStatus::Failed,
                            message,
                            Some(crate::react::llm_error_kind(kind)),
                        )
                        .await?,
                    ));
                }
                return Err(err);
            }
        };
        let Message::Assistant {
            content,
            tool_calls,
        } = completion.message
        else {
            return Err(Error::Loop(
                "completion message was not an assistant message".into(),
            ));
        };
        ctx.transcript.push(Message::Assistant {
            content,
            tool_calls: tool_calls.clone(),
        })?;
        if cancel.is_cancelled() {
            return Ok(PlanPhase::Stopped(
                stop(ctx, events, RunStatus::Cancelled, "cancelled", None).await?,
            ));
        }

        let parsed = match tool_calls.as_slice() {
            [call] if call.name == "submit_plan" => match &call.arguments {
                Some(value) => validate_plan(value, ctx.limits.max_fanout),
                None => Err("submit_plan arguments were not JSON".to_string()),
            },
            _ => Err("expected one submit_plan call".to_string()),
        };
        match parsed {
            Ok(plan) => return Ok(PlanPhase::Ready(plan)),
            Err(error) => {
                last_error = error;
                if attempt >= attempts {
                    return Ok(PlanPhase::Stopped(
                        stop(
                            ctx,
                            events,
                            RunStatus::Failed,
                            cut_bytes(&last_error, PLAN_ERROR_MAX),
                            Some("plan_invalid"),
                        )
                        .await?,
                    ));
                }
                ctx.transcript.push(Message::User {
                    content: repair_message(&last_error),
                })?;
            }
        }
    }
}

fn repair_message(error: &str) -> String {
    format!("{REPAIR}\n{}", cut_bytes(error, PLAN_ERROR_MAX))
}

async fn walk(
    ctx: &mut RunContext,
    events: &Arc<dyn RunEvents>,
    cancel: &CancellationToken,
    spawner: &Arc<SubagentSpawner>,
    plan: Plan,
) -> Result<RunOutcome, Error> {
    let mut steps = plan.steps;
    loop {
        if cancel.is_cancelled() {
            return stop(ctx, events, RunStatus::Cancelled, "cancelled", None).await;
        }
        poll_sessions(&ctx.sessions).await;
        apply_skips(&mut steps);
        let pending = steps
            .iter()
            .any(|step| matches!(step.status, StepPhase::Pending));
        let running = steps
            .iter()
            .any(|step| matches!(step.status, StepPhase::Running));
        if !pending && !running {
            let status = aggregate(&steps);
            return stop(
                ctx,
                events,
                status,
                aggregate_summary(&plan.goal, &steps),
                None,
            )
            .await;
        }
        let Some(members) = pick_candidate(&steps) else {
            if running {
                return Err(Error::Loop(
                    "plan walker has a running step and no outstanding join".into(),
                ));
            }
            return stop(
                ctx,
                events,
                RunStatus::Failed,
                "plan graph is stuck",
                Some("plan_invalid"),
            )
            .await;
        };
        let size = members.len();
        if spawner.live().saturating_add(size) > ctx.limits.max_live_subagents {
            return stop(
                ctx,
                events,
                RunStatus::Failed,
                "live_subagents limit",
                Some("limit"),
            )
            .await;
        }
        for index in &members {
            steps[*index].status = StepPhase::Running;
        }
        events.phase(ctx.run_id, Phase::Join).await;
        // One count per walker join, not per member. Nested tool calls stay on the step.
        events.tool_used(ctx.run_id, SPAWN_SUBAGENTS).await;

        let agents = members
            .iter()
            .map(|index| ChildSpec {
                name: steps[*index].id.clone(),
                task: step_task(&steps, *index),
                context: None,
            })
            .collect();
        // Steps stay at the walker depth. The spawner assigns this and does not add 1.
        let joined = match spawner
            .spawn_and_join(SpawnRequest {
                parent: ctx.run_id,
                child_depth: ctx.depth,
                agents,
                timeout: ctx.limits.spawn_timeout,
                failure_policy: ctx.limits.failure_policy,
                cancel: cancel.clone(),
                origin: ChildOrigin::Step,
            })
            .await
        {
            Ok(outcome) => outcome,
            Err(Error::Tool {
                kind: agentsmith_core::ToolErrorKind::Limit,
                message,
                ..
            }) => {
                for index in &members {
                    steps[*index].status = StepPhase::Pending;
                }
                return stop(ctx, events, RunStatus::Failed, message, Some("limit")).await;
            }
            Err(Error::Tool {
                kind: agentsmith_core::ToolErrorKind::Cancelled,
                ..
            }) => return stop(ctx, events, RunStatus::Cancelled, "cancelled", None).await,
            Err(err) => return Err(err),
        };
        if joined.agents.len() != members.len() {
            return Err(Error::Loop(
                "join returned a different number of children than the candidate".into(),
            ));
        }
        for (index, agent) in members.iter().zip(joined.agents) {
            steps[*index].run_id = Some(agent.run_id);
            steps[*index].summary = agent.summary;
            steps[*index].status = StepPhase::Done(agent.status);
        }
        events.phase(ctx.run_id, Phase::Plan).await;
        if cancel.is_cancelled() {
            return stop(ctx, events, RunStatus::Cancelled, "cancelled", None).await;
        }
    }
}

fn apply_skips(steps: &mut [StepExec]) {
    loop {
        let mut changed = false;
        for index in 0..steps.len() {
            if !matches!(steps[index].status, StepPhase::Pending) {
                continue;
            }
            let blocked = steps[index].depends_on.iter().any(|id| {
                match find(steps, id).map(|step| &step.status) {
                    Some(
                        StepPhase::Done(RunStatus::Succeeded)
                        | StepPhase::Pending
                        | StepPhase::Running,
                    ) => false,
                    Some(_) => true,
                    None => true,
                }
            });
            if blocked {
                steps[index].status = StepPhase::Skipped;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn pick_candidate(steps: &[StepExec]) -> Option<Vec<usize>> {
    let mut best: Option<(usize, Vec<usize>)> = None;
    for (index, step) in steps.iter().enumerate() {
        if step.parallel_group.is_none() && is_ready(steps, index) {
            consider(&mut best, index, vec![index]);
        }
    }
    let mut seen = BTreeMap::<String, ()>::new();
    for step in steps {
        let Some(group) = step.parallel_group.clone() else {
            continue;
        };
        if seen.insert(group.clone(), ()).is_some() {
            continue;
        }
        let members: Vec<usize> = steps
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.parallel_group.as_deref() == Some(group.as_str()))
            .map(|(index, _)| index)
            .collect();
        let active: Vec<usize> = members
            .iter()
            .copied()
            .filter(|index| !matches!(steps[*index].status, StepPhase::Skipped))
            .collect();
        if active.is_empty() {
            continue;
        }
        if active.iter().all(|index| is_ready(steps, *index)) {
            let index = active.iter().copied().min().unwrap_or(0);
            consider(&mut best, index, active);
        }
    }
    best.map(|(_, members)| members)
}

fn consider(best: &mut Option<(usize, Vec<usize>)>, index: usize, members: Vec<usize>) {
    match best {
        Some((best_index, _)) if *best_index <= index => {}
        _ => *best = Some((index, members)),
    }
}

fn is_ready(steps: &[StepExec], index: usize) -> bool {
    if !matches!(steps[index].status, StepPhase::Pending) {
        return false;
    }
    steps[index].depends_on.iter().all(|id| {
        matches!(
            find(steps, id).map(|step| &step.status),
            Some(StepPhase::Done(RunStatus::Succeeded))
        )
    })
}

fn find<'a>(steps: &'a [StepExec], id: &str) -> Option<&'a StepExec> {
    steps.iter().find(|step| step.id == id)
}

fn aggregate(steps: &[StepExec]) -> RunStatus {
    let started: Vec<&StepExec> = steps.iter().filter(|step| step.run_id.is_some()).collect();
    let skipped = steps
        .iter()
        .any(|step| matches!(step.status, StepPhase::Skipped));
    let successes = started
        .iter()
        .filter(|step| matches!(step.status, StepPhase::Done(RunStatus::Succeeded)))
        .count();
    if !started.is_empty() && successes == started.len() && !skipped {
        RunStatus::Succeeded
    } else if successes > 0 {
        RunStatus::Partial
    } else {
        RunStatus::Failed
    }
}

fn aggregate_summary(goal: &str, steps: &[StepExec]) -> String {
    let mut lines = vec![goal.to_string()];
    for step in steps {
        lines.push(format!("{}: {}", step.id, phase_label(step)));
        if step.run_id.is_some() && !step.summary.is_empty() {
            lines.push(step.summary.clone());
        }
    }
    lines.join("\n")
}

fn phase_label(step: &StepExec) -> String {
    match step.status {
        StepPhase::Pending => "pending".to_string(),
        StepPhase::Running => "running".to_string(),
        StepPhase::Skipped => "skipped".to_string(),
        StepPhase::Done(status) => status.label().to_string(),
    }
}

fn step_task(steps: &[StepExec], index: usize) -> String {
    let step = &steps[index];
    if step.depends_on.is_empty() {
        return step.instruction.clone();
    }
    let mut out = format!("{}\n\nPrior results:\n", step.instruction);
    for id in &step.depends_on {
        let Some(dep) = find(steps, id) else {
            continue;
        };
        match dep.status {
            StepPhase::Skipped => {
                out.push_str(&format!("- {id}: skipped\n"));
            }
            StepPhase::Done(status) => {
                out.push_str(&format!("- {id}: {}\n", status.label()));
                if dep.run_id.is_some() {
                    let summary = cut_bytes(&dep.summary, SUMMARY_MAX);
                    if !summary.is_empty() {
                        out.push_str(&summary);
                        out.push('\n');
                    }
                }
            }
            StepPhase::Pending | StepPhase::Running => {}
        }
    }
    out
}

fn validate_plan(value: &Value, max_fanout: usize) -> Result<Plan, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "submit_plan arguments must be an object".to_string())?;
    if object.keys().any(|key| key != "goal" && key != "steps") {
        return Err("submit_plan has an unknown field".to_string());
    }
    let goal = object
        .get("goal")
        .and_then(Value::as_str)
        .ok_or_else(|| "goal must be a string".to_string())?;
    if goal.chars().count() > 500 {
        return Err("goal is longer than 500 characters".to_string());
    }
    let steps_value = object
        .get("steps")
        .and_then(Value::as_array)
        .ok_or_else(|| "steps must be an array".to_string())?;
    if steps_value.is_empty() || steps_value.len() > MAX_STEPS {
        return Err("steps must contain from 1 to 32 items".to_string());
    }
    let mut steps = Vec::with_capacity(steps_value.len());
    let mut seen = BTreeMap::<String, usize>::new();
    for (index, step) in steps_value.iter().enumerate() {
        let parsed = parse_step(step)?;
        if seen.insert(parsed.id.clone(), index).is_some() {
            return Err(format!("duplicate step id \"{}\"", parsed.id));
        }
        steps.push(parsed);
    }
    for step in &steps {
        if step.depends_on.len() > 16 {
            return Err(format!(
                "step \"{}\" depends_on has more than 16 entries",
                step.id
            ));
        }
        for dep in &step.depends_on {
            if !seen.contains_key(dep) {
                return Err(format!(
                    "step \"{}\" depends_on unknown id \"{dep}\"",
                    step.id
                ));
            }
        }
    }
    if let Some(group) = oversized_group(&steps, max_fanout) {
        return Err(group);
    }
    if let Some(edge) = intra_group_edge(&steps) {
        return Err(edge);
    }
    if !acyclic(&steps) {
        return Err("depends_on graph has a cycle".to_string());
    }
    Ok(Plan {
        goal: goal.to_string(),
        steps,
    })
}

fn parse_step(value: &Value) -> Result<StepExec, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "each step must be an object".to_string())?;
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "id" | "title" | "instruction" | "depends_on" | "parallel_group"
        ) {
            return Err(format!("step has an unknown field \"{key}\""));
        }
    }
    let id = required_string(object, "id")?;
    if id.is_empty() || id.chars().count() > 64 {
        return Err("step id must be from 1 to 64 characters".to_string());
    }
    let title = required_string(object, "title")?;
    if title.chars().count() > 120 {
        return Err(format!("step \"{id}\" title is longer than 120 characters"));
    }
    let instruction = required_string(object, "instruction")?;
    if instruction.chars().count() > 8000 {
        return Err(format!(
            "step \"{id}\" instruction is longer than 8000 characters"
        ));
    }
    let depends_on = match object.get("depends_on") {
        None => Vec::new(),
        Some(Value::Array(items)) => {
            let mut deps = Vec::with_capacity(items.len());
            for item in items {
                let dep = item
                    .as_str()
                    .ok_or_else(|| format!("step \"{id}\" depends_on entries must be strings"))?;
                deps.push(dep.to_string());
            }
            deps
        }
        Some(_) => return Err(format!("step \"{id}\" depends_on must be an array")),
    };
    let parallel_group = match object.get("parallel_group") {
        None => None,
        Some(Value::String(group)) => {
            if group.chars().count() > 64 {
                return Err(format!(
                    "step \"{id}\" parallel_group is longer than 64 characters"
                ));
            }
            Some(group.clone())
        }
        Some(_) => return Err(format!("step \"{id}\" parallel_group must be a string")),
    };
    Ok(StepExec {
        id,
        instruction,
        depends_on,
        parallel_group,
        status: StepPhase::Pending,
        run_id: None,
        summary: String::new(),
    })
}

fn required_string(object: &serde_json::Map<String, Value>, key: &str) -> Result<String, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("{key} must be a string"))
}

fn oversized_group(steps: &[StepExec], max_fanout: usize) -> Option<String> {
    let mut counts = BTreeMap::<String, usize>::new();
    for step in steps {
        if let Some(group) = &step.parallel_group {
            *counts.entry(group.clone()).or_insert(0) += 1;
        }
    }
    counts.into_iter().find_map(|(group, count)| {
        (count > max_fanout).then(|| {
            format!("parallel_group \"{group}\" has {count} members, max_fanout is {max_fanout}")
        })
    })
}

fn intra_group_edge(steps: &[StepExec]) -> Option<String> {
    for (start, step) in steps.iter().enumerate() {
        let Some(group) = step.parallel_group.as_deref() else {
            continue;
        };
        // Follow depends_on only. An edge between every group member would
        // reject a legal parallel group as a cycle.
        let Some(hit) = reaches_group_member(steps, start, group) else {
            continue;
        };
        if step.depends_on.iter().any(|dep| dep == &hit) {
            return Some(format!(
                "depends_on \"{hit}\" stays inside parallel_group \"{group}\""
            ));
        }
        return Some(format!(
            "step \"{}\" reaches \"{hit}\" inside parallel_group \"{group}\"",
            step.id
        ));
    }
    None
}

fn reaches_group_member(steps: &[StepExec], start: usize, group: &str) -> Option<String> {
    let mut stack = Vec::new();
    for dep in &steps[start].depends_on {
        if let Some(index) = steps.iter().position(|candidate| candidate.id == *dep) {
            stack.push(index);
        }
    }
    let mut seen = vec![false; steps.len()];
    while let Some(index) = stack.pop() {
        if seen[index] {
            continue;
        }
        seen[index] = true;
        if steps[index].parallel_group.as_deref() == Some(group) {
            return Some(steps[index].id.clone());
        }
        for dep in &steps[index].depends_on {
            if let Some(next) = steps.iter().position(|candidate| candidate.id == *dep) {
                stack.push(next);
            }
        }
    }
    None
}

fn acyclic(steps: &[StepExec]) -> bool {
    let index_of: BTreeMap<&str, usize> = steps
        .iter()
        .enumerate()
        .map(|(index, step)| (step.id.as_str(), index))
        .collect();
    let mut color = vec![0u8; steps.len()];
    for start in 0..steps.len() {
        if !visit(start, steps, &index_of, &mut color) {
            return false;
        }
    }
    true
}

fn visit(
    index: usize,
    steps: &[StepExec],
    index_of: &BTreeMap<&str, usize>,
    color: &mut [u8],
) -> bool {
    if color[index] == 1 {
        return false;
    }
    if color[index] == 2 {
        return true;
    }
    color[index] = 1;
    for dep in &steps[index].depends_on {
        let Some(next) = index_of.get(dep.as_str()).copied() else {
            return false;
        };
        if !visit(next, steps, index_of, color) {
            return false;
        }
    }
    color[index] = 2;
    true
}

fn submit_plan_definition() -> ToolDefinition {
    ToolDefinition {
        name: "submit_plan".to_string(),
        description: "Submit the plan for this task. This is the only tool on the planning turn."
            .to_string(),
        parameters: json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["goal", "steps"],
            "properties": {
                "goal": { "type": "string", "maxLength": 500 },
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["id", "title", "instruction"],
                        "properties": {
                            "id": { "type": "string", "minLength": 1, "maxLength": 64 },
                            "title": { "type": "string", "maxLength": 120 },
                            "instruction": { "type": "string", "maxLength": 8000 },
                            "depends_on": {
                                "type": "array",
                                "maxItems": 16,
                                "items": { "type": "string" }
                            },
                            "parallel_group": { "type": "string", "maxLength": 64 }
                        }
                    }
                }
            }
        }),
    }
}

async fn stop(
    ctx: &mut RunContext,
    events: &Arc<dyn RunEvents>,
    status: RunStatus,
    summary: impl Into<String>,
    error_kind: Option<&str>,
) -> Result<RunOutcome, Error> {
    let (status, error_kind) = events
        .finished(ctx.run_id, status, error_kind.map(str::to_string))
        .await;
    if let Err(err) = ctx
        .transcript
        .finish(status, error_kind.as_deref(), 0, Phase::Plan)
    {
        eprintln!(
            "agentsmith: jsonl finish append failed for run {}: {err}",
            ctx.run_id
        );
    }
    Ok(RunOutcome {
        status,
        summary: summary.into(),
        error_kind,
        transcript: std::mem::take(&mut ctx.transcript).into_messages(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(id: &str, deps: &[&str], group: Option<&str>) -> StepExec {
        StepExec {
            id: id.to_string(),
            instruction: format!("do {id}"),
            depends_on: deps.iter().map(|dep| (*dep).to_string()).collect(),
            parallel_group: group.map(str::to_string),
            status: StepPhase::Pending,
            run_id: None,
            summary: String::new(),
        }
    }

    #[test]
    fn skipped_dependency_contributes_status_and_no_summary() {
        let mut steps = vec![step("a", &[], None), step("b", &["a"], None)];
        steps[0].status = StepPhase::Skipped;
        steps[0].summary = "secret transcript".into();
        let task = step_task(&steps, 1);
        assert!(task.contains("Prior results"));
        assert!(task.contains("- a: skipped"));
        assert!(!task.contains("secret transcript"));
        assert!(!task.contains("tool"));
    }

    #[test]
    fn ran_dependency_includes_a_capped_summary() {
        let mut steps = vec![step("a", &[], None), step("b", &["a"], None)];
        steps[0].status = StepPhase::Done(RunStatus::Succeeded);
        steps[0].run_id = Some(uuid::Uuid::nil());
        steps[0].summary = "é".repeat(3000);
        let task = step_task(&steps, 1);
        assert!(task.contains("- a: succeeded"));
        assert!(task.is_char_boundary(task.len()));
        let summary = task.split("- a: succeeded\n").nth(1).unwrap().trim_end();
        assert!(summary.len() <= SUMMARY_MAX, "{}", summary.len());
        assert!(summary.chars().count() < 3000);
        assert_eq!(summary.len() % "é".len(), 0);
    }

    #[test]
    fn oversized_group_and_intra_group_edge_are_plan_invalid() {
        let big = json!({
            "goal": "g",
            "steps": (0..3).map(|index| json!({
                "id": format!("s{index}"),
                "title": "t",
                "instruction": "i",
                "parallel_group": "g"
            })).collect::<Vec<_>>()
        });
        let Err(err) = validate_plan(&big, 2) else {
            panic!("expected an oversized group to be rejected");
        };
        assert!(err.contains("max_fanout"), "{err}");

        let edge = json!({
            "goal": "g",
            "steps": [
                {"id": "a", "title": "t", "instruction": "i", "parallel_group": "g"},
                {"id": "b", "title": "t", "instruction": "i", "parallel_group": "g", "depends_on": ["a"]}
            ]
        });
        let Err(err) = validate_plan(&edge, 8) else {
            panic!("expected an intra-group edge to be rejected");
        };
        assert!(err.contains("parallel_group"), "{err}");

        let indirect = json!({
            "goal": "g",
            "steps": [
                {"id": "c", "title": "t", "instruction": "i", "parallel_group": "g"},
                {"id": "b", "title": "t", "instruction": "i", "depends_on": ["c"]},
                {"id": "a", "title": "t", "instruction": "i", "parallel_group": "g", "depends_on": ["b"]}
            ]
        });
        let Err(err) = validate_plan(&indirect, 8) else {
            panic!("expected an indirect group dependency to be rejected");
        };
        assert!(err.contains("parallel_group"), "{err}");

        let leaving = json!({
            "goal": "g",
            "steps": [
                {"id": "a", "title": "t", "instruction": "i"},
                {"id": "b", "title": "t", "instruction": "i", "parallel_group": "g", "depends_on": ["a"]},
                {"id": "c", "title": "t", "instruction": "i", "parallel_group": "g", "depends_on": ["a"]}
            ]
        });
        assert!(validate_plan(&leaving, 8).is_ok());
    }
}
