use clap::{arg, ArgMatches, Command};

pub(crate) fn create_engineer_command() -> Command {
    let command = Command::new("engineer")
        .about("Engineer agent")
        .arg(arg!([NAME]));
    command
}


pub(crate) fn handle_engineer_command(matches: ArgMatches, command: &ArgMatches) {

}