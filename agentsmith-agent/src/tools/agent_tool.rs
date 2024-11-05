
use serde_json::Value;
use agentsmith_common::error::error::SystemResult;
use crate::tools::tool::{SimpleToolExecution, ToolType};

#[derive(Clone)]
pub struct CallAgentTool {
    pub r#type: ToolType,
    pub code: String,
    pub description: String,
    pub url: String,
    pub method: String,
    pub headers: String,
    pub input_schema: Value,
}


impl SimpleToolExecution for CallAgentTool {

    async fn execute(&self, input: &Value) -> SystemResult<Value> {
        todo!()
    }
}