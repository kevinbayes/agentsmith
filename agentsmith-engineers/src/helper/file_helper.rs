use std::{fs, io};
use std::io::Read;

pub fn read_string_file(file_path: &str) -> Result<String, io::Error> {
    // Read the YAML file
    let mut file = fs::File::open(file_path)?;

    println!("Reading string from {}.", file_path);

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    Ok(contents)
}