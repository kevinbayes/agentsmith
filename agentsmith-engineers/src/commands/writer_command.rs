use clap::{arg, ArgMatches, Command};

pub(crate) fn create_writer_command() -> Command {
    let command = Command::new("writer")
        .about("Writer agent")
        .arg(arg!([NAME]));
    command
}


pub(crate) fn handle_writer_command(matches: ArgMatches, command: &ArgMatches) {

}