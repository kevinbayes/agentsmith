use serde_json::Value;
use agentsmith_common::error::error::{SystemError, SystemResult};
use crate::tools::agent_tool::CallAgentTool;
use crate::tools::web_tool::SimpleJsonWebClientTool;

#[derive(Clone,)]
pub enum Tool {
    CallAgentTool(CallAgentTool),
    SimpleJsonWebClientTool(SimpleJsonWebClientTool),
}

#[derive(Clone, Debug)]
pub enum ToolType {
    Agent,
    Function,
}

#[derive(Clone, Debug)]
pub struct ToolResult {
    pub id: String,
    pub code: String,
    pub value: Value,
}

pub trait SimpleToolExecution {

    async fn execute(&self, id: Option<String>, input: &Value) -> SystemResult<ToolResult>;

    fn format_result(&self, result_value: Value) -> Vec<String> {
        println!("Tool result recording: {:?}", result_value);
        if result_value.is_null() {
            vec![]
        } else if result_value.is_object() {
            let mut messages: Vec<String> = vec![];

            if let Some(result) = result_value.get("result") {
                messages.push(result.as_str().unwrap().to_string());
            }

            if let Some(result) = result_value.get("body") {
                messages.push(result.clone().to_string());
            }

            if let Some(error) = result_value.get("error") {
                messages.push(error.as_str().unwrap().to_string());
            }

            messages
        } else {

            vec![result_value.to_string()]
        }
    }
}

impl SimpleToolExecution for Tool {

    #[allow(unused)]
    async fn execute(&self, id: Option<String>, input: &Value) -> SystemResult<ToolResult> {
        match self {
            Tool::CallAgentTool(tool) => {
                tool.execute(id, input).await
            },
            Tool::SimpleJsonWebClientTool(tool) => {
                tool.execute(id, input).await
            },
            _ => Err(SystemError::ToolError { id: 0, code: 1})
        }
    }
}
