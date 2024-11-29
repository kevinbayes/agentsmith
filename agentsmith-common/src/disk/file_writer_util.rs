use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use crate::error::error::{SystemError, SystemResult};

pub fn open_file(path: &str) -> SystemResult<File> {

    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| {
            println!("Error creating file {}.", e);
            SystemError::MemoryError { code: 1000, id: 0 }
        })?;

    Ok(file)
}

pub fn write_to_disk(path: &str, content: &str) -> SystemResult<bool> {

    let mut file: File = open_file(path).expect("File not found.");

    file.write(content.as_bytes()).unwrap();

    Ok(true)
}
