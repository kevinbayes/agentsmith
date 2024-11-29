use async_trait::async_trait;
use agentsmith_agent::agent::agent::Agent;
use agentsmith_common::error::error::SystemResult;
use crate::runtime::agents::agent::{AgentExecution, SweTeamAgent};
use crate::runtime::local::SoftwareProject;

pub struct LeadAgent {
    pub id: String,
    pub agent: Agent,
    pub team: Vec<Box<SweTeamAgent>>,
}

impl LeadAgent {
    async fn execute(&mut self, project: &mut SoftwareProject) -> SystemResult<()> {
        todo!()
    }
}
