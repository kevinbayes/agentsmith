use std::{fs, io};
use std::collections::HashMap;
use std::io::Read;
use serde::Deserialize;
use agentsmith_common::config::config::{RedisConfig as CRedisConfig};
use agentsmith_common::config::config::{DatabaseConfig as CDatabaseConfig};
use agentsmith_common::config::config::{OAuthConfig as COAuthConfig};

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
    pub qdrant: QdrantConfig,
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

impl Into<COAuthConfig> for OAuthConfig {
    fn into(self) -> COAuthConfig {
        COAuthConfig {
            audience: self.audience.clone(),
            jwks_domain: self.jwks_domain.clone(),
            jwks_path: self.jwks_path.clone(),
            jwks_protocol: self.jwks_protocol.clone(),
        }
    }
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


impl Into<CRedisConfig> for RedisConfig {

    fn into(self) -> CRedisConfig {
        CRedisConfig {
            host: self.host.clone(),
            connection_pool_size: self.connection_pool_size.clone(),
        }
    }
}


#[derive(Clone, Debug, Deserialize)]
pub struct QdrantConfig {
    // Define your configuration structure
    pub host: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DatabaseConfig {
    // Define your configuration structure
    pub connection: String,
}

impl Into<CDatabaseConfig> for DatabaseConfig {
    fn into(self) -> CDatabaseConfig {
        CDatabaseConfig {
            connection: self.connection.clone(),
        }
    }
}