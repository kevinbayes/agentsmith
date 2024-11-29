use clap::Error;

pub(crate) fn handle_hello() -> Result<String, Error> {
    println!("Hello!");
    Ok(String::from("hello"))
}