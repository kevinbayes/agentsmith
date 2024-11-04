use serde_json::Value;

#[derive(Clone)]
pub struct AgentTool {
    pub code: String,
    pub r#type: String,
}