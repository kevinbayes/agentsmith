use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ArangoConfig {
    // Define your configuration structure
    pub protocol: String,
    pub host: String,
    pub port: String,
    pub user: String,
    pub pass: String,
}