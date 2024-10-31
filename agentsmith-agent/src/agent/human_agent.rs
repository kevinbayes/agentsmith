use std::sync::Arc;
use crate::agent::agent_tool::AgentTool;
use crate::llm::llm_factory::LLM;
use crate::memory::memory::Memory;

#[derive(Clone)]
pub struct HumanAgent {
    pub id: String,
    pub name: String,
    pub tenant: String,
    pub config: crate::agent::agent::AgentConfig,
}