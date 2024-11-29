use clap::{arg, ArgMatches, Command};

pub(crate) fn create_architect_command() -> Command {
    let command = Command::new("architect")
        .about("Architect agent")
        .arg(arg!([NAME]));
    command
}


pub(crate) fn handle_architect_command(matches: ArgMatches, command: &ArgMatches) {

}