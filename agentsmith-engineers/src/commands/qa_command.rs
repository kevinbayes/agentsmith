use clap::{arg, ArgMatches, Command};

pub(crate) fn create_qa_command() -> Command {
    let command = Command::new("qa")
        .about("QA agent")
        .arg(arg!([NAME]));
    command
}


pub(crate) fn handle_qa_command(matches: ArgMatches, command: &ArgMatches) {

}