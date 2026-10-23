use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use agentsmith_core::redact::redact;
use agentsmith_core::Error;
use agentsmith_llm::OpenAiCompatible;
use agentsmith_mcp::{McpSupervisor, McpSupervisorConfig};
use agentsmith_runtime::{
    run_foreground, CancellationToken, ForegroundParams, LoopMode, RunEvents, RunStatus, Supervisor,
};
use agentsmith_tools::{SessionPoll, ToolRegistry};

use super::{init_tracing, reporter};

pub fn execute(
    config_path: Option<&Path>,
    task: &str,
    mode: Option<LoopMode>,
    spec_path: Option<&Path>,
    workdir: Option<&Path>,
) -> Result<ExitCode, Error> {
    let booted = Instant::now();
    let config = agentsmith_core::config::load(config_path)?;
    init_tracing(config.harness.log_level);
    let loaded = match spec_path {
        Some(path) => Some(agentsmith_runtime::load_spec(path)?),
        None => None,
    };
    // The synthetic spec is not a mode. Omitting both flags stays an error.
    let resolved = agentsmith_runtime::resolve_mode(mode, loaded.as_ref())?;
    let spec = loaded.unwrap_or_else(agentsmith_runtime::AgentSpec::react_default);
    let model = resolve_model(&spec, &config.llm.model)?;
    if config.llm.api_key.is_empty() {
        return Err(Error::Config("llm.api_key is required to run".into()));
    }
    let workdir = workdir.unwrap_or(config.harness.workdir.as_path());
    let workdir = workdir
        .canonicalize()
        .map_err(|err| Error::Config(format!("workdir {}: {err}", workdir.display())))?;
    let llm = OpenAiCompatible::new(
        &config.llm.base_url,
        config.llm.api_key.clone(),
        Duration::from_secs(config.llm.timeout_secs),
        u32::try_from(config.llm.retries).unwrap_or(u32::MAX),
    )?;

    let cancel = CancellationToken::new();
    let mcp_servers = config.mcp.servers.clone();
    let max_output_bytes = config.tools.max_output_bytes;
    let shutdown_grace = Duration::from_secs(config.service.shutdown_grace_secs);
    let reporting = reporter::activate(&config);
    let (registry, mcp) = if mcp_servers.is_empty() {
        let registry = reporting.then(ToolRegistry::new);
        (registry, None)
    } else {
        let registry = ToolRegistry::new();
        let mcp = McpSupervisor::new(McpSupervisorConfig {
            registry: registry.clone(),
            servers: mcp_servers,
            max_output_bytes,
            shutdown_grace,
            cancel: cancel.clone(),
        });
        (Some(registry), Some(mcp))
    };
    let sessions = mcp.clone().map(|mcp| mcp as Arc<dyn SessionPoll>);
    // A required connect failure drops `ready` unsent, so nothing is posted.
    let (events, ready, pending) = if reporting {
        let registry = registry
            .clone()
            .expect("an enabled reporter shares the tool registry");
        let supervisor = Arc::new(Supervisor::new());
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let events = Some(Arc::clone(&supervisor) as Arc<dyn RunEvents>);
        let pending = PendingReport {
            config: config.clone(),
            supervisor,
            registry,
            ready: ready_rx,
            booted,
            grace: shutdown_grace,
        };
        (events, Some(ready_tx), Some(pending))
    } else {
        (None, None, None)
    };
    let outcome = block_on(
        run_foreground(ForegroundParams {
            task: task.to_string(),
            spec,
            config,
            model,
            workdir,
            llm: Arc::new(llm),
            cancel: cancel.clone(),
            mode: resolved,
            registry,
            sessions,
            events,
            ready,
        }),
        cancel,
        mcp,
        pending,
    )?;

    if !outcome.summary.is_empty() {
        println!("{}", outcome.summary);
    }
    Ok(exit_code(outcome.status))
}

pub(crate) fn resolve_model(
    spec: &agentsmith_runtime::AgentSpec,
    config_model: &str,
) -> Result<String, Error> {
    if let Some(model) = &spec.model {
        return Ok(model.clone());
    }
    if config_model.is_empty() {
        return Err(Error::Config("llm.model is required to run".into()));
    }
    Ok(config_model.to_string())
}

fn exit_code(status: RunStatus) -> ExitCode {
    match status {
        RunStatus::Succeeded => ExitCode::SUCCESS,
        RunStatus::Partial | RunStatus::MaxIterations => ExitCode::from(2),
        RunStatus::Cancelled => ExitCode::from(130),
        // A root run has no deadline. `timed_out` is only a join target.
        RunStatus::Failed | RunStatus::TimedOut | RunStatus::Running => ExitCode::from(1),
    }
}

struct PendingReport {
    config: agentsmith_core::config::Config,
    supervisor: Arc<Supervisor>,
    registry: ToolRegistry,
    ready: tokio::sync::oneshot::Receiver<()>,
    booted: Instant,
    grace: Duration,
}

fn block_on(
    run: impl std::future::Future<Output = Result<agentsmith_runtime::RunOutcome, Error>>,
    cancel: CancellationToken,
    mcp: Option<Arc<McpSupervisor>>,
    pending: Option<PendingReport>,
) -> Result<agentsmith_runtime::RunOutcome, Error> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let signal_cancel = cancel.clone();
        tokio::spawn(async move {
            cancel_on_signal(signal_cancel).await;
        });
        let mut run = std::pin::pin!(run);
        let mut reporter = None;
        let mut finished = None;
        let grace = pending.as_ref().map(|pending| pending.grace);
        if let Some(pending) = pending {
            let PendingReport {
                config,
                supervisor,
                registry,
                mut ready,
                booted,
                grace: _,
            } = pending;
            // A run can send `ready` and finish inside one poll. The biased
            // branch would miss that signal, so check the channel again.
            // The run arm completes the future; polling it again panics.
            let signaled = tokio::select! {
                biased;
                result = &mut ready => result.is_ok(),
                result = &mut run => {
                    let signaled = ready.try_recv().is_ok();
                    finished = Some(result);
                    signaled
                }
            };
            if signaled {
                reporter = reporter::start(reporter::Parts {
                    config,
                    mode: agentsmith_protocol::ProcessMode::Foreground,
                    supervisor,
                    registry,
                    mcp: mcp.clone(),
                    booted,
                });
            }
        }
        let result = match finished {
            Some(result) => result,
            None => run.await,
        };
        if let Some(reporter) = reporter {
            let deadline = Instant::now() + grace.unwrap_or_default();
            reporter.shutdown(deadline).await;
            let remaining = deadline.saturating_duration_since(Instant::now());
            shutdown_mcp(mcp, Some(remaining)).await;
        } else {
            shutdown_mcp(mcp, None).await;
        }
        result
    })
}

async fn shutdown_mcp(mcp: Option<Arc<McpSupervisor>>, within: Option<Duration>) {
    let Some(mcp) = mcp else {
        return;
    };
    let result = match within {
        Some(grace) => mcp.shutdown_within(grace).await,
        None => mcp.shutdown().await,
    };
    if let Err(err) = result {
        eprintln!("agentsmith: mcp shutdown: {}", redact(&err.to_string()));
    }
}

pub(crate) async fn cancel_on_signal(cancel: CancellationToken) {
    #[cfg(unix)]
    {
        let mut sigterm =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(signal) => signal,
                Err(_) => {
                    let _ = tokio::signal::ctrl_c().await;
                    cancel.cancel();
                    return;
                }
            };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = sigterm.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
    cancel.cancel();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_model_overrides_config_and_empty_model_is_an_error() {
        let spec = agentsmith_runtime::AgentSpec::react_default();
        assert_eq!(resolve_model(&spec, "gpt").unwrap(), "gpt");
        assert!(resolve_model(&spec, "")
            .unwrap_err()
            .to_string()
            .contains("llm.model"));
        let spec = agentsmith_runtime::AgentSpec {
            model: Some("spec-model".into()),
            ..agentsmith_runtime::AgentSpec::react_default()
        };
        assert_eq!(resolve_model(&spec, "gpt").unwrap(), "spec-model");
    }

    #[test]
    fn partial_exits_2_and_timed_out_exits_1() {
        assert_eq!(exit_code(RunStatus::Partial), ExitCode::from(2));
        assert_eq!(exit_code(RunStatus::MaxIterations), ExitCode::from(2));
        assert_eq!(exit_code(RunStatus::TimedOut), ExitCode::from(1));
        assert_eq!(exit_code(RunStatus::Failed), ExitCode::from(1));
        assert_eq!(exit_code(RunStatus::Cancelled), ExitCode::from(130));
    }
}
