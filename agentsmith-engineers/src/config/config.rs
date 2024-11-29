use std::{fs, io};
use std::collections::HashMap;
use std::io::Read;
use serde_derive::Deserialize;
use agentsmith_agent::memory::memory::MemoryConfiguration;
use agentsmith_common::config::config::ServerConfig;

pub fn read_swe_config(file_path: &str) -> Result<SweConfig, io::Error> {
    // Read the YAML file
    let mut file = fs::File::open(file_path)?;

    println!("Reading configuration from {}.", file_path);

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    // Parse the YAML contents
    let config: SweConfig = serde_json::from_str(&contents)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    Ok(config)
}

#[derive(Clone, Debug, Deserialize)]
pub struct SweConfig {
    pub sweagents: SweAgentsConfig,
    pub config: ServerConfig,
}


#[derive(Clone, Debug, Deserialize)]
pub struct SweAgentsConfig {
    pub prompt_directory: String,
    pub lead: SweAgentConfig,
    pub architect: SweAgentConfig,
    pub engineer: SweAgentConfig,
    pub reviewer: SweAgentConfig,
    pub qa: SweAgentConfig,
    pub writer: SweAgentConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SweAgentConfig {
    // Define your configuration structure
    pub vendor: String,
    pub role: String,
    pub model: String,
    pub r#type: String,
    pub memory: MemoryConfiguration,
}
