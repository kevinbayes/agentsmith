use serde_derive::{Deserialize, Serialize};
use agentsmith_agent::agent::agent::Agent;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SoftwareProjectAudit {
    pub id: String,
    pub interaction_id: String,
    pub agent: String,
    pub summary: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SoftwareProject {
    pub name: String,
    pub description: String,
    pub requirement: String,
    pub conventions: SoftwareConventions,
    pub source: SoftwareSourceCode,
    pub tasks: Vec<ProjectTask>,
    pub audit_trail: Vec<SoftwareProjectAudit>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SoftwareConventions {
    pub language: String,
    pub libraries: String,
    pub description: String,
    pub examples: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SoftwareSourceCode {
    pub description: String,
    pub path: String,
    pub repository: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProjectTask {
    pub name: String,
    pub description: String,
}


#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn test_architect_agent() {

    }
}