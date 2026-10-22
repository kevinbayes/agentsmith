use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use agentsmith_core::Error;
use agentsmith_llm::OpenAiCompatible;
use agentsmith_runtime::{
    run_foreground, CancellationToken, ForegroundParams, LoopMode, RunStatus,
};

pub fn execute(
    config_path: Option<&Path>,
    task: &str,
    mode: Option<LoopMode>,
    spec_path: Option<&Path>,
    workdir: Option<&Path>,
) -> Result<ExitCode, Error> {
    let config = agentsmith_core::config::load(config_path)?;
    let loaded = match spec_path {
        Some(path) => Some(agentsmith_runtime::load_spec(path)?),
        None => None,
    };
    // The synthetic spec is not a mode. Omitting both flags stays an error.
    let resolved = agentsmith_runtime::resolve_mode(mode, loaded.as_ref())?;
    if resolved != LoopMode::React {
        return Err(Error::Config("plan mode is not in this build".into()));
    }
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
    let outcome = block_on(
        run_foreground(ForegroundParams {
            task: task.to_string(),
            spec,
            config,
            model,
            workdir,
            llm: Arc::new(llm),
            cancel: cancel.clone(),
        }),
        cancel,
    )?;

    if !outcome.summary.is_empty() {
        println!("{}", outcome.summary);
    }
    Ok(exit_code(outcome.status))
}

fn resolve_model(
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
        RunStatus::MaxIterations => ExitCode::from(2),
        RunStatus::Cancelled => ExitCode::from(130),
        RunStatus::Failed | RunStatus::Partial | RunStatus::TimedOut | RunStatus::Running => {
            ExitCode::from(1)
        }
    }
}

fn block_on(
    run: impl std::future::Future<Output = Result<agentsmith_runtime::RunOutcome, Error>>,
    cancel: CancellationToken,
) -> Result<agentsmith_runtime::RunOutcome, Error> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let signal_cancel = cancel.clone();
        tokio::spawn(async move {
            cancel_on_signal(signal_cancel).await;
        });
        run.await
    })
}

async fn cancel_on_signal(cancel: CancellationToken) {
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
}
