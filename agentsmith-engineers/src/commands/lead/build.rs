use std::fs;
use std::fs::File;
use std::io::prelude::*;
use clap::{ArgMatches, Error};
use minijinja::{context, path_loader, Environment};
use crate::config::config::SweConfig;
use crate::helper::template::build_template_environment;

pub(crate) fn handle_build(configuration: &SweConfig, specification_dir: ArgMatches, command: &ArgMatches, sub_command: &ArgMatches) -> Result<String, Error> {
    println!("Building lead...");

    let specification_location = sub_command.get_one::<String>("specification").expect("Must have a specification.");

    handle_build_internal(configuration, specification_location, command, sub_command)
}

pub(crate) fn handle_build_internal(configuration: &SweConfig, specification_location: &String, command: &ArgMatches, sub_command: &ArgMatches) -> Result<String, Error> {

    let instructions = read_instructions(specification_location)?;

    let prompt_path = configuration.sweagents.prompt_directory.clone();
    let initial_prompt = compile_initial_prompt(&prompt_path, &instructions)?;

    Ok(String::from("test"))
}

fn read_instructions(path: &String) -> Result<String, Error> {
    let mut file = File::open(path)?;
    let mut instructions = String::new();
    file.read_to_string(&mut instructions)?;
    println!("Instructions: \n {}", instructions.clone());
    Ok(instructions)
}

fn compile_initial_prompt(path: &String, instructions: &String) -> Result<String, Error> {

    let mut env = build_template_environment(path);

    let template = env.get_template("software_lead/initial.prompt.jinja").unwrap();
    let context = context! {
        name => "World",
        instructions => instructions.clone(),
    };
    let rendered_string = template.render(&context).unwrap();
    println!("Initial prompt: \n {}", rendered_string.clone());

    Ok(rendered_string)
}

