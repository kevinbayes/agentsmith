use crate::llm::llm_factory::LLM;

#[derive(Clone)]
pub struct SoftwareReviewerAgent {
    id: String,
    name: String,
    config: crate::agent::agent::AgentConfig,
    llm: LLM,
    environment: crate::agent::agent::AgentEnvironment
}