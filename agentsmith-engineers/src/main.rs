mod config;
mod helper;
mod runtime;
mod commands;

use std::path::PathBuf;
use clap::{arg, command, value_parser, Arg, ArgAction, Command};
use log::{debug, info};
use agentsmith_common::config::config::read_config;
use crate::commands::lead_command::{create_lead_command, handle_lead_command};
use crate::commands::engineer_command::{create_engineer_command, handle_engineer_command};
use crate::commands::architect_command::create_architect_command;
use crate::commands::reviewer_command::{create_reviewer_command, handle_reviewer_command};
use crate::commands::qa_command::{create_qa_command, handle_qa_command};
use crate::commands::writer_command::{create_writer_command, handle_writer_command};
use crate::commands::cli_command::{create_cli_command, handle_cli_command};
use crate::config::config::read_swe_config;

fn main() {

    let matches = Command::new("AgentSmith Engineers")
        .version("1.0")
        .about("Your augmenting engineering team!")
        .next_line_help(true)
        .arg(Arg::new("working-directory").long("working-dir").default_value("../"))
        .arg(Arg::new("configuration").long("config").default_value("./sample-config.json"))
        .subcommand_required(true)
        .subcommand(create_lead_command())
        .subcommand(create_architect_command())
        .subcommand(create_engineer_command())
        .subcommand(create_reviewer_command())
        .subcommand(create_qa_command())
        .subcommand(create_writer_command())
        .subcommand(create_cli_command())
        .get_matches();

    let Some(config) = matches.get_one::<String>("configuration") else { panic!("Must have a config location"); };

    let configuration = read_swe_config(config).expect("Failed to read config");

    match matches.clone().subcommand() {
        Some(("cli", cli_command)) => {
            debug!("enter: cli");
            handle_cli_command(matches, cli_command)
        },
        Some(("lead", orchestrator_command)) => {
            debug!("enter: lead");
            handle_lead_command(configuration, matches, orchestrator_command)
        },
        Some(("engineer", engineer_command)) => {
            debug!("enter: engineer");
            handle_engineer_command(matches, engineer_command)
        },
        Some(("reviewer", reviewer_command)) => {
            debug!("enter: reviewer");
            handle_reviewer_command(matches, reviewer_command)
        },
        Some(("qa", qa_command)) => {
            debug!("enter: qa");
            handle_qa_command(matches, qa_command)
        },
        Some(("writer", writer_command)) => {
            debug!("enter: writer");
            handle_writer_command(matches, writer_command)
        },
        _ => {
            println!("Unknown command");
        }
    }
}
