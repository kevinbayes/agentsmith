use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

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
}

#[derive(Debug, Subcommand)]
enum ConfigAction {
    /// Load config and print its 12-character fingerprint.
    Check,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Config {
            action: ConfigAction::Check,
        } => commands::config::check(cli.config.as_deref()),
    };
    if let Err(err) = result {
        eprintln!("error: {err}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
