use serde::{Deserialize, Serialize};
use crate::agent::agent::AgentConfig;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct AgentSpecification {
    pub id: String,
    pub name: String,
    pub role: String,
    pub description: String,
    pub tags: Vec<String>,
    pub r#type: String,
    pub configuration: AgentConfig,
}