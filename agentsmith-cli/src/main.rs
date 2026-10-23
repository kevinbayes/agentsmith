use std::path::PathBuf;
use std::process::ExitCode;

use agentsmith_core::Error;
use clap::{Parser, Subcommand, ValueEnum};

mod commands;

#[derive(Debug, Parser)]
#[command(name = "agentsmith", version, about = "Lightweight agent harness")]
struct Cli {
    /// TOML config file. Overrides AGENTSMITH_CONFIG and the search path.
    #[arg(long, global = true, value_name = "PATH")]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Inspect and validate configuration.
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Run one task in the foreground, or start the loopback service.
    Run {
        /// Task text for the agent.
        #[arg(long, conflicts_with = "service")]
        task: Option<String>,
        /// Long-running loopback service. Does not start a task.
        #[arg(long, conflicts_with_all = ["task", "mode", "spec", "workdir"])]
        service: bool,
        /// Loop mode. Required unless the spec sets mode.
        #[arg(long, value_enum)]
        mode: Option<ModeArg>,
        /// Agent spec TOML.
        #[arg(long, value_name = "PATH")]
        spec: Option<PathBuf>,
        /// Working directory. Defaults to harness.workdir.
        #[arg(long, value_name = "PATH")]
        workdir: Option<PathBuf>,
    },
    /// Install the systemd unit and seed a config if one is absent.
    Install {
        /// Install the user unit. This is the default.
        #[arg(long, conflicts_with = "system")]
        user: bool,
        /// Install the system unit.
        #[arg(long, conflicts_with = "user")]
        system: bool,
        /// Enable the unit without starting it.
        #[arg(long)]
        no_start: bool,
    },
    /// Remove the systemd unit. `--purge` also deletes the data directory.
    Uninstall {
        /// Remove the user unit. This is the default.
        #[arg(long, conflicts_with = "system")]
        user: bool,
        /// Remove the system unit.
        #[arg(long, conflicts_with = "user")]
        system: bool,
        /// Delete the data directory after the unit is gone.
        #[arg(long)]
        purge: bool,
    },
}

#[derive(Debug, Subcommand)]
enum ConfigAction {
    /// Load config and print its 12-character fingerprint.
    Check,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModeArg {
    /// ReAct tool loop.
    React,
    /// Plan, then execute each step.
    Plan,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Config {
            action: ConfigAction::Check,
        } => match commands::config::check(cli.config.as_deref()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::from(1)
            }
        },
        Command::Run {
            task,
            service,
            mode,
            spec,
            workdir,
        } => {
            if service {
                return match commands::service::execute(cli.config.as_deref()) {
                    Ok(code) => code,
                    Err(err) => fail(err),
                };
            }
            let Some(task) = task else {
                eprintln!("error: run requires --task or --service");
                return ExitCode::from(2);
            };
            let mode = mode.map(|value| match value {
                ModeArg::React => agentsmith_runtime::LoopMode::React,
                ModeArg::Plan => agentsmith_runtime::LoopMode::Plan,
            });
            match commands::run::execute(
                cli.config.as_deref(),
                &task,
                mode,
                spec.as_deref(),
                workdir.as_deref(),
            ) {
                Ok(code) => code,
                Err(err) => fail(err),
            }
        }
        Command::Install {
            user: _,
            system,
            no_start,
        } => match commands::install::execute_install(system, no_start) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => fail(err),
        },
        Command::Uninstall {
            user: _,
            system,
            purge,
        } => match commands::install::execute_uninstall(system, purge) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => fail(err),
        },
    }
}

fn fail(err: Error) -> ExitCode {
    // The non-Linux guard is specified as this exact sentence, with no prefix.
    if err.to_string().contains(commands::install::LINUX_ONLY) {
        eprintln!("{}", commands::install::LINUX_ONLY);
        return ExitCode::from(2);
    }
    eprintln!("error: {err}");
    match err {
        Error::Config(_) => ExitCode::from(2),
        _ => ExitCode::from(1),
    }
}
