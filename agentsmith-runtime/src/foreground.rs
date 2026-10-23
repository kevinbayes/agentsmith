use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use agentsmith_core::config::{Config, ToolsConfig};
use agentsmith_core::Error;
use agentsmith_tools::{register_builtins, BuiltinLimits, SessionPoll, ToolRegistry};

use crate::loop_::{LoopLimits, RunContext, RunEvents, RunOutcome};
use crate::plan::PlanExecuteLoop;
use crate::react::ReactLoop;
use crate::run_tools::{RunKind, RunTools, ToolFilter};
use crate::spawn::{SpawnerConfig, SubagentSpawner};
use crate::spec::{AgentSpec, LoopMode};
use crate::supervisor::Supervisor;
use crate::transcript::Transcript;
use crate::CancellationToken;

pub struct ForegroundParams {
    pub task: String,
    pub spec: AgentSpec,
    pub config: Config,
    /// Model already resolved from the spec, or `llm.model` when the spec model is empty.
    pub model: String,
    pub workdir: PathBuf,
    pub llm: Arc<dyn agentsmith_llm::Llm>,
    pub cancel: CancellationToken,
    /// Explicit mode already won over the spec, when the caller passed one.
    pub mode: LoopMode,
    /// Shared with the poller when MCP servers are configured. `None` builds a fresh map.
    pub registry: Option<ToolRegistry>,
    /// `None` when no MCP servers are configured. Nothing is connected in that case.
    pub sessions: Option<Arc<dyn SessionPoll>>,
}

pub async fn run_foreground(params: ForegroundParams) -> Result<RunOutcome, Error> {
    let registry = match &params.registry {
        Some(registry) => registry.clone(),
        None => ToolRegistry::new(),
    };
    register_foreground_builtins(&registry, &params.config.tools)?;
    let sessions = params.sessions.clone();
    let max_output_tokens = u32::try_from(params.config.llm.max_output_tokens)
        .map_err(|_| Error::Config("llm.max_output_tokens does not fit in u32".into()))?;
    let limits = LoopLimits::from_config(&params.config.loop_)?;
    let events: Arc<dyn RunEvents> = Arc::new(Supervisor::new());
    let data_dir = params.config.harness.data_dir.clone();
    let allow = params.config.tools.allow.clone();
    // Install before the root view is resolved so the intersection can drop it.
    let spawner = SubagentSpawner::new(SpawnerConfig {
        registry: registry.clone(),
        llm: Arc::clone(&params.llm),
        model: params.model.clone(),
        max_output_tokens,
        temperature: params.config.llm.temperature,
        max_output_bytes: params.config.tools.max_output_bytes,
        workdir: params.workdir.clone(),
        data_dir: data_dir.clone(),
        spec: params.spec.clone(),
        allow: allow.clone(),
        limits: limits.clone(),
        events: Arc::clone(&events),
        sessions: sessions.clone(),
    });
    spawner.install()?;
    // Required connect failure must not open a transcript or start the loop.
    if let Some(sessions) = &sessions {
        sessions.poll().await?;
    }
    if let Some(names) = params.spec.subagent_tools.as_deref() {
        // The step view, after the depth drop. A spawned view at depth 1 still
        // lists spawn_subagents when max_subagent_depth is 0.
        let step_view = RunTools::resolve_for(
            registry.clone(),
            ToolFilter {
                allow: allow.as_deref(),
                spec_tools: params.spec.tools.as_deref(),
                subagent_tools: None,
                kind: RunKind::Step,
                depth: 0,
                max_subagent_depth: limits.max_subagent_depth,
            },
        )?;
        for name in names {
            if step_view.get(name).is_none() {
                return Err(Error::Config(format!(
                    "spec.subagent_tools includes \"{name}\", which is not available"
                )));
            }
        }
    }
    let tools = RunTools::resolve(registry, allow.as_deref(), params.spec.tools.as_deref())?;
    let run_id = agentsmith_core::id::new_id();
    let transcript = Transcript::with_log(&data_dir, run_id).map_err(|err| match err {
        Error::Io(err) => Error::Config(format!("data_dir: {err}")),
        other => other,
    })?;
    let max_iterations = match params.mode {
        LoopMode::React => limits.react_max_iterations,
        LoopMode::Plan => limits.step_max_iterations,
    };
    let name = params.spec.name.clone();
    let ctx = RunContext {
        run_id,
        parent_run_id: None,
        depth: 0,
        name,
        task: params.task,
        extra_context: None,
        model: params.model,
        max_output_tokens,
        temperature: params.config.llm.temperature,
        max_output_bytes: params.config.tools.max_output_bytes,
        max_iterations,
        workdir: params.workdir,
        llm: params.llm,
        tools,
        limits,
        cancel: params.cancel,
        events,
        transcript,
        spec: params.spec,
        spawner: Some(spawner),
        sessions,
    };
    match params.mode {
        LoopMode::React => ReactLoop.run(ctx).await,
        LoopMode::Plan => PlanExecuteLoop.run(ctx).await,
    }
}

/// `tools.shell.enabled == None` enables shell for this foreground run.
fn register_foreground_builtins(registry: &ToolRegistry, tools: &ToolsConfig) -> Result<(), Error> {
    let enable_shell = tools.shell.enabled.unwrap_or(true);
    register_builtins(
        registry,
        &BuiltinLimits {
            max_read_bytes: tools.files.max_read_bytes,
            max_write_bytes: tools.files.max_write_bytes,
            max_output_bytes: tools.max_output_bytes,
            shell_timeout: Duration::from_secs(tools.shell.timeout_secs),
            shell_max_timeout: Duration::from_secs(tools.shell.max_timeout_secs),
            enable_shell,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_shell_is_on_and_explicit_false_is_off() {
        let enabled = ToolRegistry::new();
        let mut tools = ToolsConfig::default();
        tools.shell.enabled = None;
        register_foreground_builtins(&enabled, &tools).unwrap();
        assert!(enabled.names().iter().any(|name| name == "shell"));

        let disabled = ToolRegistry::new();
        tools.shell.enabled = Some(false);
        register_foreground_builtins(&disabled, &tools).unwrap();
        assert!(!disabled.names().iter().any(|name| name == "shell"));

        let forced = ToolRegistry::new();
        tools.shell.enabled = Some(true);
        register_foreground_builtins(&forced, &tools).unwrap();
        assert!(forced.names().iter().any(|name| name == "shell"));
    }
}
