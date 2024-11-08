use std::collections::HashMap;
use std::iter::Map;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use agentsmith_common::error::error::SystemError;
use crate::agent::agent::Agent;
use crate::tools::registry::{SafeToolRegistry, ToolRegistry};
use crate::tools::tool::{SimpleToolExecution, Tool as ActualTool, ToolResult};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Prompt {
    Simple { system: String, user: String, tools: Option<Vec<Tool>>, tool_choice: Option<ToolChoice> },
    Messages { system: String, messages: Vec<PromptMessage>, tools: Option<Vec<Tool>>, tool_choice: Option<ToolChoice> }
}

impl Prompt {
    pub fn new_simple(system: String, user: String) -> Self {
        Self::Simple { system, user, tools: None, tool_choice: None }
    }
    pub fn new_simple_with_tools(system: String, user: String, tool_choice: Option<ToolChoice>, tools: Vec<Tool>) -> Self {
        Self::Simple { system, user, tools: Some(tools), tool_choice }
    }
    pub fn new_message(system: String, messages: Vec<PromptMessage>, tool_choice: ToolChoice, tools: Vec<Tool>, ) -> Self {
        Self::Messages { system, messages, tools: Some(tools), tool_choice: Some(tool_choice) }
    }
    pub fn new_message_for_agent(agent: &Agent, messages: Vec<PromptMessage>, tool_registry: &SafeToolRegistry) -> Self {

        let tools: Vec<Tool> = agent.toolbox()
            .iter()
            .filter_map(|x| tool_registry.read().unwrap().get_tool(&x.code))
            .map(|x| x.as_ref().clone().into())
            .collect();

        let tool_choice = agent.tool_choice().clone();

        let tools = if !tools.is_empty() {
            Some(tools)
        } else {
            None
        };

        Self::Messages { system: agent.system_prompt(), messages, tools, tool_choice }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PromptMessage {
    System {
        role: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        role: String,
        content: Vec<UserContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        role: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<Vec<AssistantContent>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<AssistantToolCall>,
    },
    Tool { role: String, content: Vec<String>, name: String },
}

impl PromptMessage {

    pub fn from_tool_result(tool_result: &ToolResult, tool: &ActualTool) -> Self {

        Self::Tool {
            name: tool_result.code.clone(),
            content: tool.format_result(tool_result.value.clone()),
            role: "tool".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AssistantContent {
    Text {
        #[serde(rename = "type")]
        type_: String,
        text: String,
    },
    Image {
        #[serde(rename = "type")]
        type_: String,
        content_type: Option<String>,
        image_url: ContentImageUrl,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContentImageUrl {
    pub url: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text {
        #[serde(rename = "type")]
        type_: String,
        text: String,
    },
    Image {
        #[serde(rename = "type")]
        type_: String,
        content_type: Option<String>,
        image_url: ContentImageUrl,
    },
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssistantToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub function: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Auto {
        #[serde(rename = "type")]
        type_: String,
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        #[serde(rename = "type")]
        type_: String,
        disable_parallel_tool_use: Option<bool>,
    },
    Required {
        #[serde(rename = "type")]
        type_: String,
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        #[serde(rename = "type")]
        type_: String,
        name: String,
        disable_parallel_tool_use: Option<bool>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    pub r#type: Option<String>,
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}


impl Into<Tool> for ActualTool {

    fn into(self) -> Tool {

        match self {
            crate::tools::tool::Tool::CallAgentTool(tool) => {
                Tool {
                    r#type: Some(format!("{:?}", tool.r#type)),
                    description: tool.description.clone(),
                    name: tool.code.clone(),
                    input_schema: tool.input_schema.clone(),
                }
            },
            crate::tools::tool::Tool::SimpleJsonWebClientTool(tool) => {
                Tool {
                    r#type: Some(format!("{:?}", tool.r#type)),
                    description: tool.description.clone(),
                    name: tool.code.clone(),
                    input_schema: tool.input_schema.clone(),
                }
            },
        }
    }
}