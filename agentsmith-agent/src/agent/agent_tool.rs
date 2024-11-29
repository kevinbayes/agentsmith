use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct AgentTool {
    pub code: String,
    pub r#type: String,
}