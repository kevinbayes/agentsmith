use std::sync::Arc;

use agentsmith_core::{Error, LlmErrorKind};
use agentsmith_llm::{
    CompletionRequest, Message, StopReason, ToolCall, ToolChoice, ToolDefinition,
};
use agentsmith_tools::ToolCtx;

use crate::loop_::{poll_sessions, Phase, RunContext, RunEvents, RunMeta, RunOutcome, RunStatus};
use crate::run_tools::SPAWN_SUBAGENTS;
use crate::spec::LoopMode;

#[derive(Debug, Default)]
pub struct ReactLoop;

impl ReactLoop {
    pub async fn run(&self, ctx: RunContext) -> Result<RunOutcome, Error> {
        react(ctx).await
    }
}

async fn react(mut ctx: RunContext) -> Result<RunOutcome, Error> {
    let events = Arc::clone(&ctx.events);
    ctx.transcript
        .arm_unwind(Arc::clone(&events), ctx.run_id, Phase::React);
    let llm = Arc::clone(&ctx.llm);
    let cancel = ctx.cancel.clone();

    events
        .started(
            ctx.run_id,
            RunMeta {
                parent_run_id: ctx.parent_run_id,
                depth: ctx.depth,
                name: ctx.name.clone(),
                max_iterations: ctx.max_iterations,
                phase: Phase::React,
                mode: LoopMode::React,
                spec_id: ctx.spec.id.clone(),
                task: ctx.task.clone(),
            },
        )
        .await;
    events.phase(ctx.run_id, Phase::React).await;
    events.iteration(ctx.run_id, 0).await;
    // Held until this run returns so spawn_subagents can read depth and the prompt.
    let _caller = ctx
        .spawner
        .clone()
        .map(|spawner| spawner.enter(ctx.run_id, ctx.depth, ctx.spec.system_prompt.clone()));

    ctx.transcript.push(Message::System {
        content: ctx.spec.system_prompt.clone(),
    })?;
    ctx.transcript.push(Message::User {
        content: user_text(&ctx.task, ctx.extra_context.as_deref()),
    })?;

    let max_iters = ctx.max_iterations;
    let max_errors = ctx.limits.max_consecutive_tool_errors;
    let mut iteration = 0u32;
    let mut consecutive_errors = 0u32;

    loop {
        if cancel.is_cancelled() {
            return stop(
                &mut ctx,
                &events,
                RunStatus::Cancelled,
                "cancelled",
                None,
                iteration,
            )
            .await;
        }
        poll_sessions(&ctx.sessions).await;
        // A cap of zero means the run cannot take a model turn.
        if max_iters == 0 {
            return stop(
                &mut ctx,
                &events,
                RunStatus::MaxIterations,
                "max iterations",
                Some("max_iterations"),
                iteration,
            )
            .await;
        }

        let request = CompletionRequest {
            model: ctx.model.clone(),
            messages: ctx.transcript.messages().to_vec(),
            tools: ctx.tools.definitions().into_iter().map(llm_tool).collect(),
            tool_choice: ToolChoice::Auto,
            max_output_tokens: ctx.max_output_tokens,
            temperature: ctx.temperature,
            cancel: cancel.clone(),
        };

        let completion = match llm.complete(request).await {
            Ok(completion) => completion,
            Err(err) => {
                // Cancel is the first row of the stop table, ahead of an LLM error.
                if cancel.is_cancelled() || matches!(err, Error::Cancelled) {
                    return stop(
                        &mut ctx,
                        &events,
                        RunStatus::Cancelled,
                        "cancelled",
                        None,
                        iteration,
                    )
                    .await;
                }
                if let Error::Llm { kind, message } = err {
                    return stop(
                        &mut ctx,
                        &events,
                        RunStatus::Failed,
                        message,
                        Some(llm_error_kind(kind)),
                        iteration,
                    )
                    .await;
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
        let stop_reason = completion.stop;
        ctx.transcript.push(Message::Assistant {
            content: content.clone(),
            tool_calls: tool_calls.clone(),
        })?;

        if cancel.is_cancelled() {
            return stop(
                &mut ctx,
                &events,
                RunStatus::Cancelled,
                "cancelled",
                None,
                iteration,
            )
            .await;
        }

        if tool_calls.is_empty() {
            return match stop_reason {
                StopReason::Length => {
                    stop(
                        &mut ctx,
                        &events,
                        RunStatus::Failed,
                        text_or(&content, "length"),
                        Some("length"),
                        iteration,
                    )
                    .await
                }
                StopReason::ContentFilter => {
                    stop(
                        &mut ctx,
                        &events,
                        RunStatus::Failed,
                        text_or(&content, "content_filter"),
                        Some("content_filter"),
                        iteration,
                    )
                    .await
                }
                StopReason::EndTurn | StopReason::ToolUse => {
                    stop(
                        &mut ctx,
                        &events,
                        RunStatus::Succeeded,
                        content.unwrap_or_default(),
                        None,
                        iteration,
                    )
                    .await
                }
            };
        }

        let truncation_note = match stop_reason {
            StopReason::Length => {
                Some("The previous assistant message was truncated (finish_reason=length).")
            }
            StopReason::ContentFilter => {
                Some("The previous assistant message was truncated (finish_reason=content_filter).")
            }
            StopReason::EndTurn | StopReason::ToolUse => None,
        };

        for call in tool_calls {
            if cancel.is_cancelled() {
                return stop(
                    &mut ctx,
                    &events,
                    RunStatus::Cancelled,
                    "cancelled",
                    None,
                    iteration,
                )
                .await;
            }
            let ToolCall {
                id,
                name,
                arguments,
                ..
            } = call;
            let (observation, finish_answer, errored) = match arguments {
                None => {
                    consecutive_errors = consecutive_errors.saturating_add(1);
                    (
                        observation_err("invalid_args", "invalid tool arguments"),
                        None,
                        true,
                    )
                }
                Some(_) if ctx.tools.get(&name).is_none() => {
                    consecutive_errors = consecutive_errors.saturating_add(1);
                    // `mcp_down` is only a server the supervisor marked down.
                    // A re-list that drops a name leaves the server up.
                    let observation = if ctx.tools.mcp_down(&name) {
                        observation_err("mcp_down", "mcp server down")
                    } else {
                        observation_err("not_found", format!("no such tool: {name}"))
                    };
                    (observation, None, true)
                }
                Some(arguments) => {
                    events.tool_used(ctx.run_id, &name).await;
                    let tool_ctx = ToolCtx {
                        run_id: ctx.run_id,
                        workdir: ctx.workdir.clone(),
                        cancel: cancel.clone(),
                    };
                    match ctx.tools.call(&name, tool_ctx, arguments).await {
                        Ok(output) if name == SPAWN_SUBAGENTS => {
                            // The tool already returned the join object. Do not wrap it.
                            // `ok: false` counts toward the consecutive-error cap.
                            let ok = serde_json::from_str::<serde_json::Value>(&output.content)
                                .ok()
                                .and_then(|value| value.get("ok").and_then(|flag| flag.as_bool()))
                                .unwrap_or(false);
                            if ok {
                                consecutive_errors = 0;
                            } else {
                                consecutive_errors = consecutive_errors.saturating_add(1);
                            }
                            (output.content, None, !ok)
                        }
                        Ok(output) => {
                            consecutive_errors = 0;
                            let (observation, capped) =
                                observation_ok(&output.content, ctx.max_output_bytes);
                            let answer = (name == "finish").then_some(capped);
                            (observation, answer, false)
                        }
                        Err(Error::Tool { kind, message, .. }) => {
                            consecutive_errors = consecutive_errors.saturating_add(1);
                            (observation_err(&kind.to_string(), message), None, true)
                        }
                        Err(Error::Cancelled) => {
                            consecutive_errors = consecutive_errors.saturating_add(1);
                            (observation_err("cancelled", "cancelled"), None, true)
                        }
                        Err(err) => {
                            consecutive_errors = consecutive_errors.saturating_add(1);
                            (observation_err("execution", err.to_string()), None, true)
                        }
                    }
                }
            };
            ctx.transcript.push(Message::Tool {
                tool_call_id: id,
                content: observation,
            })?;

            if cancel.is_cancelled() {
                return stop(
                    &mut ctx,
                    &events,
                    RunStatus::Cancelled,
                    "cancelled",
                    None,
                    iteration,
                )
                .await;
            }
            if let Some(answer) = finish_answer {
                return stop(
                    &mut ctx,
                    &events,
                    RunStatus::Succeeded,
                    answer,
                    None,
                    iteration,
                )
                .await;
            }
            // A success resets the counter to 0. With a cap of 0, `0 >= 0` would
            // fail that success, so the cap applies only to an error observation.
            if errored && consecutive_errors >= max_errors {
                return stop(
                    &mut ctx,
                    &events,
                    RunStatus::Failed,
                    "consecutive tool errors",
                    Some("tool_errors"),
                    iteration,
                )
                .await;
            }
        }

        if let Some(note) = truncation_note {
            ctx.transcript.push(Message::User {
                content: note.to_string(),
            })?;
        }

        iteration = iteration.saturating_add(1);
        events.iteration(ctx.run_id, iteration).await;
        if iteration == max_iters {
            return stop(
                &mut ctx,
                &events,
                RunStatus::MaxIterations,
                "max iterations",
                Some("max_iterations"),
                iteration,
            )
            .await;
        }
    }
}

fn user_text(task: &str, extra: Option<&str>) -> String {
    match extra {
        Some(extra) if !extra.is_empty() => format!("{task}\n\n{extra}"),
        _ => task.to_string(),
    }
}

fn text_or(content: &Option<String>, fallback: &str) -> String {
    content
        .clone()
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| fallback.to_string())
}

fn llm_tool(def: agentsmith_tools::ToolDefinition) -> ToolDefinition {
    ToolDefinition {
        name: def.name,
        description: def.description,
        parameters: def.parameters,
    }
}

pub(crate) fn llm_error_kind(kind: LlmErrorKind) -> &'static str {
    match kind {
        LlmErrorKind::Auth => "auth",
        LlmErrorKind::BadRequest => "bad_request",
        LlmErrorKind::Unavailable => "unavailable",
        LlmErrorKind::Timeout => "timeout",
    }
}

async fn stop(
    ctx: &mut RunContext,
    events: &Arc<dyn RunEvents>,
    status: RunStatus,
    summary: impl Into<String>,
    error_kind: Option<&str>,
    iteration: u32,
) -> Result<RunOutcome, Error> {
    let (status, error_kind) = events
        .finished(ctx.run_id, status, error_kind.map(str::to_string))
        .await;
    if let Err(err) = ctx
        .transcript
        .finish(status, error_kind.as_deref(), iteration, Phase::React)
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

fn observation_ok(content: &str, max_bytes: u64) -> (String, String) {
    let content = cap_content(content, max_bytes);
    let json = serde_json::json!({ "ok": true, "content": content }).to_string();
    (json, content)
}

fn observation_err(kind: &str, message: impl Into<String>) -> String {
    serde_json::json!({
        "ok": false,
        "error": { "kind": kind, "message": message.into() }
    })
    .to_string()
}

const TRUNCATED_SUFFIX: &str = "\n[truncated]";

fn cap_content(content: &str, max_bytes: u64) -> String {
    let max_bytes = usize::try_from(max_bytes).unwrap_or(usize::MAX);
    if content.len() <= max_bytes {
        return content.to_string();
    }
    if max_bytes >= TRUNCATED_SUFFIX.len() {
        let end = floor_char_boundary(content, max_bytes - TRUNCATED_SUFFIX.len());
        let mut out = String::with_capacity(end + TRUNCATED_SUFFIX.len());
        out.push_str(&content[..end]);
        out.push_str(TRUNCATED_SUFFIX);
        return out;
    }
    let end = floor_char_boundary(content, max_bytes);
    content[..end].to_string()
}

/// `str::floor_char_boundary` is Rust 1.91. This crate's MSRV is 1.88.
fn floor_char_boundary(content: &str, max_bytes: usize) -> usize {
    let mut end = max_bytes.min(content.len());
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_content_cuts_on_a_char_boundary() {
        let text = "éééééééé";
        assert_eq!(cap_content(text, 14), "é\n[truncated]");
        assert_eq!(cap_content(text, 13), "\n[truncated]");
        assert_eq!(cap_content("ok", 14), "ok");
        assert!(cap_content(text, 1).is_char_boundary(cap_content(text, 1).len()));
    }
}
