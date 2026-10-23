use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use agentsmith_core::config::{Config, ToolsConfig};
use agentsmith_core::Error;
use agentsmith_tools::{register_builtins, BuiltinLimits, ToolRegistry};

use crate::loop_::{LoopLimits, RunContext, RunOutcome};
use crate::react::ReactLoop;
use crate::run_tools::RunTools;
use crate::spec::AgentSpec;
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
}

pub async fn run_foreground(params: ForegroundParams) -> Result<RunOutcome, Error> {
    let registry = ToolRegistry::new();
    register_foreground_builtins(&registry, &params.config.tools)?;
    let tools = RunTools::resolve(
        registry,
        params.config.tools.allow.as_deref(),
        params.spec.tools.as_deref(),
    )?;
    let max_output_tokens = u32::try_from(params.config.llm.max_output_tokens)
        .map_err(|_| Error::Config("llm.max_output_tokens does not fit in u32".into()))?;
    let limits = LoopLimits::from_config(&params.config.loop_)?;
    let run_id = agentsmith_core::id::new_id();
    let data_dir = params.config.harness.data_dir.clone();
    let transcript = Transcript::with_log(&data_dir, run_id).map_err(|err| match err {
        Error::Io(err) => Error::Config(format!("data_dir: {err}")),
        other => other,
    })?;
    let ctx = RunContext {
        run_id,
        task: params.task,
        extra_context: None,
        model: params.model,
        max_output_tokens,
        temperature: params.config.llm.temperature,
        max_output_bytes: params.config.tools.max_output_bytes,
        workdir: params.workdir,
        llm: params.llm,
        tools,
        limits,
        cancel: params.cancel,
        events: Arc::new(Supervisor::new()),
        transcript,
        spec: params.spec,
    };
    ReactLoop.run(ctx).await
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
