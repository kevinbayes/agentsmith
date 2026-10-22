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
    /// Run one task in the foreground.
    Run {
        /// Task text for the agent.
        #[arg(long)]
        task: String,
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
            mode,
            spec,
            workdir,
        } => {
            let mode = mode.map(|value| match value {
                ModeArg::React => agentsmith_runtime::LoopMode::React,
            });
            match commands::run::execute(
                cli.config.as_deref(),
                &task,
                mode,
                spec.as_deref(),
                workdir.as_deref(),
            ) {
                Ok(code) => code,
                Err(err) => {
                    eprintln!("error: {err}");
                    match err {
                        Error::Config(_) => ExitCode::from(2),
                        _ => ExitCode::from(1),
                    }
                }
            }
        }
    }
}
