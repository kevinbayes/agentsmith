use std::cmp::PartialEq;
use crate::config::config::SweConfig;
use crate::runtime::agents::agent::{build_agent_config, build_prompt_string, AgentExecution, AgentExecutionResult, AgentState};
use crate::runtime::local::{ProjectTask, SoftwareProject};
use agentsmith_agent::agent::agent::Agent;
use agentsmith_agent::agent::agent_factory::AgentFactory;
use agentsmith_common::error::error::{SystemError, SystemResult};
use minijinja::context;
use uuid::Uuid;
use agentsmith_agent::llm::llm::LLMResult;
use agentsmith_agent::llm::prompt::{Prompt, PromptMessage, UserContent};
use agentsmith_common::disk::file_writer_util::write_to_disk;
use regex::{Regex, RegexBuilder};
use serde_json::Value;
use crate::helper::command_line::get_user_input;

#[derive(Clone, Debug)]
enum ArchitectAgentState {
    Ready, Reflection, Reviewing, Packaging, Done
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

    fn ask_user_to_review(&self, message: &str) -> String {
        let question = format!("'{}'. Select (1) Accept (2) Try again (3) Abort.", message);
        let user_input = get_user_input(question.as_str());
        match user_input.as_str() {
            "1" | "2" | "3" => user_input,
            _ => {
                println!("Not a valid input please try again...");
                self.ask_user_to_review(message)
            }
        }
    }
}

fn extract_tasks_from_document(document: &str) -> SystemResult<Vec<ProjectTask>> {
    // Regex to find JSON block within ```json or <backlog> tags
    let json_regex = RegexBuilder::new(r"(?:<backlog>\n```|<backlog>\n```json|```json)(.*?)+(?:```\n</backlog>|```)").dot_matches_new_line(true).build()
        .map_err(|e| {
            println!("Error with regex: {}", e);
            SystemError::ParsingError { id: 2, code: 3000 }
        })?;

    // Find the first match
    let json_str = json_regex
        .captures(document)
        .and_then(|cap| cap.get(1))
        .map(|m| m.as_str().trim())
        .map(|s| {
            let first_index = s.find('{').unwrap_or(0);
            let last_index = s.rfind('}').map(|c| c+1).unwrap_or(s.len());
            &s[first_index..last_index]
        })
        .ok_or("No JSON block found")
        .map_err(|e| {
            println!("Error with regex: {}", e);
            SystemError::ParsingError { id: 2, code: 3001 }
        })?;

    println!("Looking at json {}.", json_str);

    // Parse the JSON
    let json_value: Value = serde_json::from_str(json_str)
        .map_err(|e| {
            println!("Error with regex: {}", e);
            SystemError::ParsingError { id: 2, code: 3002 }
        })?;

    // Extract the tasks array
    let tasks_array = json_value
        .get("tasks")
        .and_then(|v| v.as_array())
        .ok_or("Could not find tasks array")
        .map_err(|e| {
            println!("Error with regex: {}", e);
            SystemError::ParsingError { id: 2, code: 3003 }
        })?;

    // Map the tasks to ProjectTask structs
    let tasks: Vec<ProjectTask> = tasks_array
        .iter()
        .map(|task| ProjectTask {
            name: task.get("task")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            description: task.get("description")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
        })
        .collect();

    Ok(tasks)
}

impl AgentExecution for ArchitectAgent {

    async fn execute(&mut self, project: &mut SoftwareProject) -> SystemResult<AgentExecutionResult> {

        let mut latest_response: String = String::from("");

        while self.state != AgentState::Done {

            match self.state {
                AgentState::Ready => {
                    let system_template_name = format!("{}/{:?}.system.prompt.jinja", Self::ROLE, self.internal_state).to_lowercase();
                    let user_template_name = format!("{}/{:?}.user.prompt.jinja", Self::ROLE, self.internal_state).to_lowercase();

                    let context = context! {
                        project => &project,
                    };

                    let system_string = build_prompt_string(&self.prompt_path, &system_template_name, &context);
                    let user_string = build_prompt_string(&self.prompt_path, &user_template_name, &context);

                    let prompt = Prompt::new_simple(system_string, user_string,);

                    let result = self.agent.chat_completion(&prompt).await?;

                    let output_checkpoint = format!("./{}.{:?}.md", Self::ROLE, self.internal_state).to_lowercase();

                    write_to_disk(output_checkpoint.as_str(), result.message.as_str()).expect("TODO: panic message");

                    println!("{:?}", result);

                    latest_response = result.message.clone();

                    self.state = AgentState::Working;
                    self.internal_state = ArchitectAgentState::Reflection;
                }
                AgentState::Working => {

                    let template_names = match self.internal_state {
                        ArchitectAgentState::Reflection | ArchitectAgentState::Reviewing => {

                            let system_template_name = format!("{}/{:?}.system.prompt.jinja", Self::ROLE, self.internal_state).to_lowercase();
                            let user_template_name = format!("{}/{:?}.user.prompt.jinja", Self::ROLE, self.internal_state).to_lowercase();

                            (system_template_name, user_template_name)
                        }
                        _ => (String::new(), String::new())
                    };


                    let system_template_name = template_names.0;
                    let user_template_name = template_names.1;

                    if system_template_name.is_empty() {

                        self.state = AgentState::Done;
                        self.internal_state = ArchitectAgentState::Done;
                    } else {

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

                        match self.internal_state {
                            ArchitectAgentState::Reflection => {

                                self.state = AgentState::Working;
                                self.internal_state = ArchitectAgentState::Reviewing;
                            }
                            _ => {
                                let question = format!("Please review '{}'.", output_checkpoint);
                                let user_input = self.ask_user_to_review(question.as_str());
                                match user_input.as_str() {
                                    "1" => {
                                        self.state = AgentState::Completing;
                                        self.internal_state = ArchitectAgentState::Packaging;
                                    },
                                    "2" => {
                                        self.state = AgentState::Working;
                                        self.internal_state = ArchitectAgentState::Reviewing;
                                    },
                                    _ => {
                                        self.state = AgentState::Done;
                                        self.internal_state = ArchitectAgentState::Done;
                                    },
                                };
                            }
                        }
                    }
                }
                AgentState::Completing => {




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
    use crate::runtime::agents::architect_agent::{extract_tasks_from_document, ArchitectAgent};
    use crate::runtime::local::{SoftwareConventions, SoftwareProject, SoftwareSourceCode};
    use agentsmith_agent::agent::agent_factory::AgentFactory;
    use agentsmith_common::config::config::Config;
    use crate::helper::file_helper::read_string_file;

    #[tokio::test]
    async fn test_parsing_output_to_tasks_1() {

        let string = read_string_file("./resources/test/runtime/agents/architect.output.1.md").unwrap();

        let tasks = extract_tasks_from_document(&string).unwrap();

        assert_eq!(24, tasks.len());

    }

    #[tokio::test]
    async fn test_parsing_output_to_tasks_2() {

        let string = read_string_file("./resources/test/runtime/agents/architect.output.2.md").unwrap();

        let tasks = extract_tasks_from_document(&string).unwrap();

        assert_eq!(10, tasks.len());

    }

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
