use std::sync::Arc;
use agentsmith_common::error::error::{SystemError, SystemResult};
use crate::agent::agent::AgentConfig;
use crate::agent::agent_tool::AgentTool;
use crate::llm::llm_factory::LLM;
use crate::memory::memory::Memory;

#[derive(Clone)]
pub struct SimpleAgent {
    pub id: String,
    pub name: String,
    pub description: String,
    pub tenant: String,
    pub config: AgentConfig,
    pub llm: LLM,
    pub memory: Arc<Memory>,
    pub toolbox: Arc<Vec<AgentTool>>
}

impl SimpleAgent {
    pub fn new(agent_config: &AgentConfig, llm: LLM, memory: Memory) -> SystemResult<Self> {
        match agent_config.r#type.as_ref() {
            "simple" => {
                Ok(Self {
                    id: agent_config.id.clone(),
                    name: agent_config.name.clone(),
                    description: agent_config.description.clone(),
                    tenant: "".to_string(),
                    config: agent_config.clone(),
                    llm: llm,
                    toolbox: Arc::new(agent_config.toolbox.clone()),
                    memory: Arc::new(memory),
                })
            }
            _ => Err(SystemError::AgentError { id: 0, code: 0 })
        }
    }
}