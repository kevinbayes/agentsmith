use std::sync::Arc;
use serde_json::Value;
use agentsmith_common::error::error::SystemResult;
use crate::agent::agent::AgentConfig;
use crate::agent::agent_tool::AgentTool;
use crate::llm::llm_factory::LLM;
use crate::memory::memory::Memory;
use crate::tools::tool::{SimpleToolExecution, ToolType};

#[derive(Clone)]
pub struct WebClientTool {
    pub r#type: ToolType,
    pub code: String,
    pub description: String,
    pub url: String,
    pub method: String,
    pub headers: String,
    pub input_schema: Value,
    pub output_schema: Value,
}


impl SimpleToolExecution for WebClientTool {
    async fn execute(&self, input: &Value) -> SystemResult<Value> {
        todo!()
    }
}