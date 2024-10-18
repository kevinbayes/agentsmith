use std::{fs, io};
use std::collections::HashMap;
use std::io::Read;
use serde::Deserialize;

pub fn read_config(file_path: &str) -> Result<Config, io::Error> {

    let mut file = fs::File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let config: Config = serde_json::from_str(&contents)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    Ok(config)
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    // Define your configuration structure
    pub config: ServerConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ServerConfig {
    // Define your configuration structure
    pub redis: RedisConfig,
    pub database: DatabaseConfig,
    pub host: HostConfig,
    pub security: SecurityConfig,
    pub gateways: GatewaysConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GatewaysConfig {
    // Define your configuration structure
    #[serde(rename = "registry")]
    pub registry: HashMap<String, GatewayConfig>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GatewayConfig {
    #[serde(rename = "baseurl")]
    pub baseurl: String,
    #[serde(rename = "apiKey")]
    pub api_key: String,
    #[serde(rename = "model")]
    pub model: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SecurityConfig {
    // Define your configuration structure
    #[serde(rename = "oauth")]
    pub oauth: OAuthConfig,
    // Define your configuration structure
    #[serde(rename = "jwt")]
    pub jwt: SecurityJwtConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OAuthConfig {
    // Define your configuration structure
    #[serde(rename = "jwks_domain")]
    pub jwks_domain: String,
    #[serde(rename = "jwks_protocol")]
    pub jwks_protocol: String,
    #[serde(rename = "jwks_path")]
    pub jwks_path: String,
    #[serde(rename = "audience")]
    pub audience: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SecurityJwtConfig {
    // Define your configuration structure
    #[serde(rename = "secret")]
    pub secret: String,
    #[serde(rename = "issuer")]
    pub issuer: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct HostConfig {
    // Define your configuration structure
    pub host: String,
    pub port: i32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RedisConfig {
    // Define your configuration structure
    pub host: String,
    pub connection_pool_size: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DatabaseConfig {
    // Define your configuration structure
    pub connection: String,
}