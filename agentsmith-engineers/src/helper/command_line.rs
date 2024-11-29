use std::io::{stdin, stdout, Stdout};
use ratatui::crossterm::ExecutableCommand;
use ratatui::crossterm::style::{Color, ResetColor, SetForegroundColor};

pub fn get_user_input(question: &str) -> String {

    let mut stdout: Stdout = stdout();

    stdout.execute(SetForegroundColor(Color::Blue)).unwrap();
    println!("");
    println!("{}", question);

    stdout.execute(ResetColor).unwrap();

    let mut user_response: String = String::new();
    stdin()
        .read_line(&mut user_response)
        .expect("Failed to read response.");

    user_response.trim().to_string()
}