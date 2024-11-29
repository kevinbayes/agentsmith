use clap::{arg, Arg, ArgMatches, Command};
use log::debug;
use crate::config::config::{SweConfig};
use crate::commands::lead::build::handle_build;
use crate::commands::lead::hello::handle_hello;

pub(crate) fn create_lead_command() -> Command {
    let command = Command::new("lead")
        .about("lead agent").arg(arg!([NAME]))
        .subcommand_required(true)
        .subcommand(create_lead_hello_command())
        .subcommand(create_lead_build_command())
        ;
    command
}

fn create_lead_hello_command() -> Command {
    let command = Command::new("hello")
        .about("lead say hello").arg(arg!([NAME]))
        ;
    command
}

fn create_lead_build_command() -> Command {
    let command = Command::new("build")
        .about("lead build something").arg(arg!([NAME]))
        .arg(Arg::new("agents_folder").long("agents_folder").default_value("../.agents/prompts"))
        .arg(Arg::new("specification").long("specification").default_value("./specification.txt"))
        ;
    command
}


pub(crate) fn handle_lead_command(configuration: SweConfig, matches: ArgMatches, command: &ArgMatches) {

    match command.clone().subcommand() {
        Some(("hello", sub_command)) => {
            handle_hello().expect("How did I get here!");
        }
        Some(("build", sub_command)) => {
            handle_build(&configuration, matches, command, sub_command).expect("How did I get here!");
        }
        _ => {
            debug!("Unknown command")
        }
    }

}