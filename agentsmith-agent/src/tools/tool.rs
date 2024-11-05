use serde_json::Value;
use agentsmith_common::error::error::{SystemError, SystemResult};
use crate::tools::agent_tool::CallAgentTool;
use crate::tools::web_tool::WebClientTool;

#[derive(Clone,)]
pub enum Tool {
    CallAgentTool(CallAgentTool),
    WebClientTool(WebClientTool),
}

#[derive(Clone, Debug)]
pub enum ToolType {
    Agent,
    Function,
}

pub trait SimpleToolExecution {

    async fn execute(&self, input: &Value) -> SystemResult<Value>;
}

impl SimpleToolExecution for Tool {

    #[allow(unused)]
    async fn execute(&self, input: &Value) -> SystemResult<Value> {
        match self {
            Tool::CallAgentTool(tool) => {
                tool.execute(input).await
            },
            Tool::WebClientTool(tool) => {
                tool.execute(input).await
            },
            _ => Err(SystemError::ToolError { id: 0, code: 1})
        }
    }
}
