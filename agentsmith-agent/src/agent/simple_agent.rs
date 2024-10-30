use std::sync::Arc;
use crate::agent::agent_tool::AgentTool;
use crate::llm::llm_factory::LLM;
use crate::memory::memory::Memory;

#[derive(Clone)]
pub struct TextAgent {
    id: String,
    name: String,
    tenant: String,
    config: crate::agent::agent::AgentConfig,
    llm: LLM,
    memory: Arc<Memory>,
    toolbox: Arc<Vec<AgentTool>>
}