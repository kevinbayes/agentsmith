use crate::llm::llm_factory::LLM;

#[derive(Debug, Clone)]
pub struct SoftwareOrchestratorAgent {
    id: String,
    name: String,
    config: crate::agent::agent::AgentConfig,
    llm: LLM,
    environment: crate::agent::agent::AgentEnvironment
}