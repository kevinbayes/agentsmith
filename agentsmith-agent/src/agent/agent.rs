use crate::agent::simple_agent::SimpleAgent;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use agentsmith_common::error::error::SystemResult;
use crate::agent::agent_tool::AgentTool;
use crate::agent::human_agent::HumanAgent;
use crate::llm::llm::{LLMConfiguration, LLMResult};
use crate::llm::llm_factory::{LLMClient, LLM};
use crate::llm::prompt::{Prompt, ToolChoice};
use crate::memory::memory::{Memory, MemoryConfiguration};


#[derive(Clone)]
pub enum Agent {
    SimpleAgent(SimpleAgent),
    HumanAgent(HumanAgent),
}

#[derive(Clone)]
pub struct AgentAttributes {
    pub id: String,
    pub name: String,
    pub description: String,
    pub tenant: String,
    pub config: AgentConfig,
    pub llm: Arc<LLM>,
    pub memory: Arc<Memory>,
    pub toolbox: Arc<Vec<AgentTool>>
}

pub trait AgentTraits {
    fn get_attributes(&self) -> &AgentAttributes;
    fn get_state(&self) -> &String;
    fn get_state_internal(&self) -> &String;
    fn get_memory(&self) -> Arc<Memory>;
}

impl Agent {

    pub fn id(&self) -> String {
        match self {
            Agent::SimpleAgent(agent) => agent.id.clone(),
            Agent::HumanAgent(agent) => agent.id.clone(),
        }
    }

    pub fn name(&self) -> String {
        match self {
            Agent::SimpleAgent(agent) => agent.attributes.name.clone(),
            Agent::HumanAgent(agent) => agent.attributes.name.clone(),
        }
    }

    pub fn description(&self) -> String {
        match self {
            Agent::SimpleAgent(agent) => agent.attributes.description.clone(),
            Agent::HumanAgent(agent) => agent.attributes.description.clone(),
        }
    }



    pub fn system_prompt(&self) -> String {
        match self {
            Agent::SimpleAgent(agent) => agent.attributes.config.system_prompt.clone().unwrap(),
            Agent::HumanAgent(agent) => agent.id.clone(),
        }
    }

    pub fn tool_choice(&self) -> Option<ToolChoice> {
        match self {
            Agent::SimpleAgent(agent) => Some(ToolChoice::Auto { type_: "auto".to_string(), disable_parallel_tool_use: Some(true) }),
            Agent::HumanAgent(agent) => None,
        }
    }
    pub fn toolbox(&self) -> Vec<AgentTool> {
        match self {
            Agent::SimpleAgent(agent) => agent.attributes.toolbox.to_vec(),
            Agent::HumanAgent(agent) => vec![],
        }
    }

    pub async fn chat_completion(&self, prompt: &Prompt) -> SystemResult<LLMResult> {
        match self {
            Agent::SimpleAgent(agent) => {
                let llm = agent.attributes.llm.clone();
                Ok(llm.execute(prompt).await?)
            },
            Agent::HumanAgent(agent) => {
                todo!()
            },
        }
    }
}


#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct AgentConfig {
    pub id: String,
    pub name: String,
    pub description: String,
    pub r#type: String,
    pub system_prompt: Option<String>,
    pub llm: LLMConfiguration,
    pub memory: MemoryConfiguration,
    pub toolbox: Vec<AgentTool>,
}

#[derive(Debug, Clone)]
struct AgentEnvironmentConfig {
}

pub trait Tool: Debug {
    fn debug_print(&self);
}

struct ToolBoxItem(Box<dyn Tool>);

impl Debug for ToolBoxItem {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        self.0.fmt(f) // Call the debug_print method inside Tool
    }
}

impl Clone for ToolBoxItem {
    fn clone(&self) -> Self {
        // Here, we would need the underlying tool type to implement Clone.
        // This is a limitation with trait objects, so we will assume that the
        // type inside the Box implements clone manually.
        ToolBoxItem(self.0.clone_box()) // Using a helper method (clone_box)
    }
}

trait ToolClone {
    fn clone_box(&self) -> Box<dyn Tool>;
}

// Implement ToolClone for types that implement both Tool and Clone
impl<T> ToolClone for T
where
    T: 'static + Tool + Clone,
{
    fn clone_box(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
}

// Add ToolClone to the Tool trait
impl dyn Tool {
    fn clone_box(&self) -> Box<dyn Tool> {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub struct AgentEnvironment {
    config: AgentEnvironmentConfig,
    tools: Vec<ToolBoxItem>
}
