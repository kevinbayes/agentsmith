use async_trait::async_trait;
use minijinja::context;
use serde_json::Value;
use uuid::Uuid;
use agentsmith_agent::agent::agent::{Agent, AgentConfig};
use agentsmith_agent::llm::llm::{LLMConfiguration, LLMCredentials};
use agentsmith_agent::llm::prompt::PromptMessage;
use agentsmith_agent::memory::general::GeneralMemoryConfiguration;
use agentsmith_agent::memory::memory::MemoryConfiguration;
use agentsmith_agent::memory::repository::semantic_repository::{SemanticDiskRepositoryConfiguration, SemanticMemoryConfiguration};
use agentsmith_agent::memory::repository::working_repository::{WorkingMemoryConfiguration, WorkingMemoryDiskRepositoryConfiguration};
use agentsmith_common::error::error::SystemResult;
use crate::config::config::{SweAgentConfig, SweConfig};
use crate::helper::template::build_template_environment;
use crate::runtime::agents::architect_agent::ArchitectAgent;
use crate::runtime::agents::software_engineer_agent::SoftwareEngineerAgent;
use crate::runtime::local::{SoftwareProject};

pub enum SweTeamAgent {
    Architect(ArchitectAgent),
    SoftwareEngineer(SoftwareEngineerAgent),
}


pub fn build_agent_config(name: &str, description: &str, swe_agent_config: &SweAgentConfig, config: &SweConfig) -> AgentConfig {

    let id = Uuid::new_v4().to_string();

    let llm_provider = swe_agent_config.vendor.clone();

    let specific_gateway_key = format!("{}_gateway", swe_agent_config.role);

    let gateway_registry = config.config.gateways.registry.clone();

    let llm_gateway = gateway_registry.get(specific_gateway_key.as_str())
        .or_else(|| {
            let fallback_gateway_key = format!("{}_gateway", llm_provider.clone());
            gateway_registry.get(fallback_gateway_key.as_str())
        }).expect(format!("No gateway config set. Please set for {}", name.to_string()).as_str())
        ;

    let llm_configuration: LLMConfiguration = LLMConfiguration {
        provider: llm_provider,
        base_url: Some(llm_gateway.baseurl.clone()),
        model: swe_agent_config.model.clone(),
        credentials: LLMCredentials {
            api_key: llm_gateway.api_key.clone(),
        },
        stream: None,
        seed: None,
        max_tokens: None,
        top_p: None,
        version: None,
        temperature: None,
    };

    let memory_configuration : MemoryConfiguration = MemoryConfiguration {
        r#type: "general".to_string(),
        general: Some(GeneralMemoryConfiguration {
          semantic: SemanticMemoryConfiguration {
                id: "".to_string(),
              r#type: "disk".to_string(),
              arango: None,
              disk: Some(SemanticDiskRepositoryConfiguration {
                  path: "./tmp/memory".to_string(),
              })
          },
          working: WorkingMemoryConfiguration {
              id: "".to_string(),
              r#type: "disk".to_string(),
              arango: None,
              disk: Some(WorkingMemoryDiskRepositoryConfiguration {
                  path: "./tmp/memory".to_string(),
              }),
          }
        })
    };

    AgentConfig {
        id,
        name: name.to_string(),
        description: description.to_string(),
        r#type: swe_agent_config.r#type.clone(),
        system_prompt: None,
        llm: llm_configuration,
        memory: memory_configuration,
        toolbox: vec![],
    }
}

pub fn build_prompt_string(prompt_path: &String, template: &String, context: &minijinja::Value) -> String {

    let template_environment = build_template_environment(prompt_path);

    let template_name = template.to_lowercase();
    println!("Using template {}", template_name);

    let template = template_environment
        .get_template(template_name.as_str())
        .unwrap();

    let rendered_string = template.render(context).unwrap();
    println!("Produced prompt {}", rendered_string);
    rendered_string
}

#[derive(Clone, Debug)]
pub struct AgentExecutionResult {
    pub history: Vec<PromptMessage>
}

#[derive(Clone, Debug, PartialEq)]
pub enum AgentState {
    Ready, Working, Done
}


pub trait AgentExecution {

    async fn execute(&mut self, project: &mut SoftwareProject) -> SystemResult<AgentExecutionResult>;
}

impl AgentExecution for SweTeamAgent {
    async fn execute(&mut self, project: &mut SoftwareProject) -> SystemResult<AgentExecutionResult> {
        match self {
            SweTeamAgent::Architect(agent) => agent.execute(project).await,
            SweTeamAgent::SoftwareEngineer(agent) => agent.execute(project).await,
        }
    }
}
