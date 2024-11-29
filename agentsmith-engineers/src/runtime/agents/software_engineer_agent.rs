use async_trait::async_trait;
use agentsmith_agent::agent::agent::Agent;
use agentsmith_common::error::error::SystemResult;
use crate::runtime::agents::agent::{AgentExecution, AgentExecutionResult};
use crate::runtime::agents::lead_agent::LeadAgent;
use crate::runtime::local::{SoftwareProject};

pub struct SoftwareEngineerAgent {
    pub id: String,
    pub agent: Agent,
}

impl AgentExecution for SoftwareEngineerAgent {
    async fn execute(&mut self, project: &mut SoftwareProject) -> SystemResult<AgentExecutionResult> {
        todo!()
    }
}