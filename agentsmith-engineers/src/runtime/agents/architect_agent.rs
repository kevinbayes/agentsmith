use std::cmp::PartialEq;
use crate::config::config::SweConfig;
use crate::helper::template::build_template_environment;
use crate::runtime::agents::agent::{build_agent_config, build_prompt_string, AgentExecution, AgentExecutionResult, AgentState};
use crate::runtime::local::SoftwareProject;
use agentsmith_agent::agent::agent::Agent;
use agentsmith_agent::agent::agent_factory::AgentFactory;
use agentsmith_common::error::error::SystemResult;
use minijinja::context;
use uuid::Uuid;
use agentsmith_agent::llm::llm::LLMResult;
use agentsmith_agent::llm::prompt::{Prompt, PromptMessage, UserContent};
use agentsmith_common::disk::file_writer_util::write_to_disk;

#[derive(Clone, Debug)]
enum ArchitectAgentState {
    Ready, Thinking, Reviewing, Done
}

#[derive(Clone)]
pub struct ArchitectAgent {
    pub id: String,
    pub state: AgentState,
    pub internal_state: ArchitectAgentState,
    pub agent: Agent,
    pub prompt_path: String,
}

impl ArchitectAgent {
    const ROLE: &'static str = "architect";

    pub async fn new(agent_factory: &AgentFactory, config: &SweConfig) -> SystemResult<Self> {
        let id = Uuid::new_v4().to_string();
        let description = "The solution architect.";

        let agent_config =
            build_agent_config(Self::ROLE, description, &config.sweagents.architect, config);

        let agent = agent_factory.instance(agent_config).await?;

        Ok(Self {
            id,
            agent,
            state: AgentState::Ready,
            internal_state: ArchitectAgentState::Ready,
            prompt_path: config.sweagents.prompt_directory.clone(),
        })
    }

    pub(crate) fn update_project_tasks(&self, project: &mut SoftwareProject, llm_result: &LLMResult) -> SystemResult<bool> {



        Ok(true)
    }
}

impl AgentExecution for ArchitectAgent {

    async fn execute(&mut self, project: &mut SoftwareProject) -> SystemResult<AgentExecutionResult> {

        let mut latest_response: String = String::from("");

        while self.state != AgentState::Done {

            match self.state {
                AgentState::Ready => {
                    let system_template_name = format!("{}/{:?}.system.prompt.jinja", Self::ROLE, self.state).to_lowercase();
                    let user_template_name = format!("{}/{:?}.user.prompt.jinja", Self::ROLE, self.state).to_lowercase();

                    let context = context! {
                        project => &project,
                    };

                    let system_string = build_prompt_string(&self.prompt_path, &system_template_name, &context);
                    let user_string = build_prompt_string(&self.prompt_path, &user_template_name, &context);

                    let prompt = Prompt::new_simple(system_string, user_string,);

                    let result = self.agent.chat_completion(&prompt).await?;

                    let output_checkpoint = format!("./{}.{:?}.md", Self::ROLE, self.state).to_lowercase();

                    write_to_disk(output_checkpoint.as_str(), result.message.as_str()).expect("TODO: panic message");

                    println!("{:?}", result);

                    latest_response = result.message.clone();

                    self.state = AgentState::Working;
                    self.internal_state = ArchitectAgentState::Reviewing;
                }
                AgentState::Working => {

                    let system_template_name = format!("{}/{:?}.system.prompt.jinja", Self::ROLE, self.state).to_lowercase();
                    let user_template_name = format!("{}/{:?}.user.prompt.jinja", Self::ROLE, self.state).to_lowercase();

                    let context = context! {
                        project => &project,
                        latest_response => latest_response,
                    };

                    let system_string = build_prompt_string(&self.prompt_path, &system_template_name, &context);
                    let user_string = build_prompt_string(&self.prompt_path, &user_template_name, &context);

                    let prompt = Prompt::new_simple(system_string, user_string,);

                    let result = self.agent.chat_completion(&prompt).await?;

                    let output_checkpoint = format!("./{}.{:?}.md", Self::ROLE, self.state).to_lowercase();

                    write_to_disk(output_checkpoint.as_str(), result.message.as_str()).expect("TODO: panic message");

                    println!("{:?}", result);

                    latest_response = result.message.clone();

                    self.state = AgentState::Done;
                    self.internal_state = ArchitectAgentState::Done;
                }
                AgentState::Done => {
                    self.state = AgentState::Done;
                    self.internal_state = ArchitectAgentState::Done;
                }
            }
        }

        Ok(AgentExecutionResult { history: vec![] })
    }
}

#[cfg(test)]
mod tests {
    use crate::config::config::read_swe_config;
    use crate::runtime::agents::agent::AgentExecution;
    use crate::runtime::agents::architect_agent::ArchitectAgent;
    use crate::runtime::local::{SoftwareConventions, SoftwareProject, SoftwareSourceCode};
    use agentsmith_agent::agent::agent_factory::AgentFactory;
    use agentsmith_common::config::config::Config;

    #[tokio::test]
    async fn test_architect_agent() {
        let config = read_swe_config("./secret-config.json").unwrap();
        let general_config = Config {
            config: config.config.clone(),
        };
        let agent_factory = AgentFactory::new(general_config);

        let mut architect = ArchitectAgent::new(&agent_factory, &config).await.unwrap();

        let mut project = SoftwareProject {
          name: "Echo API".to_string(),
          description: "A REST api that echos what you send in.".to_string(),
            conventions: SoftwareConventions {
                libraries: r#"rust:
* axum for api"#.to_string(),
                description: r#"Technology stack:
* rust
    * Webserver library axum."#.to_string(),
                examples: "".to_string(),
                language: "rust".to_string(),
            },
            source: SoftwareSourceCode {
                description: "".to_string(),
                path: "./tmp/software".to_string(),
                repository: "".to_string()
            },
            tasks: vec![],
            requirement: r#"We require a webserver with a single api to echo what the user sends as input as json output.
The json output must have headers, query parameters, path, method and body."#.to_string(),
            audit_trail: vec![]
        };

        let result = architect.execute(&mut project).await;

        println!("{:?}", result);
        println!("{:?}", architect.state);
    }
}
