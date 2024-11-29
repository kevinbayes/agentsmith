use clap::{arg, ArgMatches, Command};

pub(crate) fn create_reviewer_command() -> Command {
    let command = Command::new("reviewer")
        .about("Reviewer agent")
        .arg(arg!([NAME]));
    command
}


pub(crate) fn handle_reviewer_command(matches: ArgMatches, command: &ArgMatches) {

}