use std::sync::Arc;
use crate::agent::agent::AgentAttributes;
use crate::agent::agent_tool::AgentTool;
use crate::llm::llm_factory::LLM;
use crate::memory::memory::Memory;

#[derive(Clone)]
pub struct HumanAgent {
    pub id: String,
    pub attributes: AgentAttributes,
}