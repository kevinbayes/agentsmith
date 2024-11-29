use clap::{arg, ArgMatches, Command};

pub(crate) fn create_cli_command() -> Command {
    let command = Command::new("cli")
        .about("Cli for engineering team")
        .arg(arg!([NAME]));
    command
}

pub(crate) fn handle_cli_command(matches: ArgMatches, command: &ArgMatches) {

}

#[cfg(test)]
mod tests {
    #[test]
    fn test_create_cli_command() {

    }
}