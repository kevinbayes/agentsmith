
use serde_json::{json, Value};
use short_uuid::ShortUuid;
use agentsmith_common::error::error::SystemResult;
use crate::tools::tool::{SimpleToolExecution, ToolResult, ToolType};

#[derive(Clone)]
pub struct CallAgentTool {
    pub r#type: ToolType,
    pub code: String,
    pub description: String,
    pub input_schema: Value,
}


impl SimpleToolExecution for CallAgentTool {

    async fn execute(&self, id: Option<String>, input: &Value) -> SystemResult<ToolResult> {
        Ok(ToolResult {
            id: id.unwrap_or(ShortUuid::generate().to_string()),
            code: self.code.clone(),
            value: input.clone(),
        })
    }
}