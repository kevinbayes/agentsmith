use std::collections::HashMap;
use agentsmith_agent::agent::agent::Agent;
use agentsmith_agent::memory::memory::{Memory, MemoryContext, RecordMemory, RetrieveMemory};
use agentsmith_agent::memory::messages::Messages;
use std::sync::Arc;
use serde_json::json;
use agentsmith_agent::llm::llm::LLMResult;
use agentsmith_agent::llm::prompt::{Prompt, PromptMessage, Tool};
use agentsmith_agent::tools::registry::{SafeToolRegistry,};
use agentsmith_agent::tools::tool::{SimpleToolExecution, ToolResult};
use agentsmith_common::error::error::SystemResult;
use uuid::{uuid, Uuid};

pub struct Swarm {
    pub memory_context: Arc<MemoryContext>,
    pub memory: Arc<Memory>,
    pub agents: Arc<Vec<Agent>>,
    pub tool_registry: Arc<SafeToolRegistry>,
    pub initial_agent: String,
    pub active_agent: String,
    pub max_turns: u16,
    pub turn: u16,
}

struct SwarmToolResults {
    next_agent: Option<String>,
    tool_results: Vec<ToolResult>,
}


pub struct SwarmResult {
    pub memory: Arc<Memory>,
}

impl Swarm {

    pub fn new(agents: Arc<Vec<Agent>>, tool_registry: &SafeToolRegistry, max_turns: u16, ) -> Self {

        if agents.is_empty() {
            panic!("No agents configured, must have at least one!");
        }

        let interaction_id = Uuid::new_v4().to_string();

        let initial_agent = agents.first().unwrap().clone().id().to_string();

        Self {
            memory_context: Arc::new(MemoryContext { interaction_id }),
            memory: Arc::new(Memory::MESSAGES(Messages::new())),
            agents: agents.clone(),
            tool_registry: Arc::new(tool_registry.clone()),
            initial_agent: initial_agent.clone(),
            active_agent: initial_agent.clone(),
            max_turns,
            turn: 0,
        }
    }

    pub fn reset(&mut self) {

        self.memory = Arc::new(Memory::MESSAGES(Messages::new()));
        self.active_agent = self.initial_agent.clone();
        self.turn = 0;
    }


    pub async fn run(&mut self, initial: PromptMessage) -> SystemResult<SwarmResult> {

        let _ = self.memory.record_prompt_messages(&self.memory_context, &vec![initial]).await;

        while self.turn < self.max_turns {

            let agent = self.agents.iter()
                .find(|item| item.id() == self.active_agent.clone())
                .unwrap();

            let messages = self.memory.retrieve_past_messages(&self.memory_context)
                .await?
                .clone();

            let prompt = Prompt::new_message_for_agent(agent, messages, &self.tool_registry,);

            let result = agent.chat_completion(&prompt).await?;

            let make_tool_calls = !result.tool_calls.is_empty();

            let assistant_message = PromptMessage::from_assistant_message(&result);
            let _ = self.memory.record_prompt_message(&self.memory_context, &assistant_message).await;

            if make_tool_calls {

                let tool_results = self.handle_function_calls(result).await;

                match tool_results.next_agent.clone() {
                    None => {}
                    Some(next_agent) => {
                        self.active_agent = next_agent;
                    }
                }

                self.append_tool_call_result_to_memory(tool_results).await;

            } else {
                self.turn = self.max_turns;
            }

            self.turn += 1;
        }

        let memory = self.memory.clone();

        Ok(SwarmResult { memory })
    }

    async fn handle_function_calls(&self, result: LLMResult) -> SwarmToolResults {

        let mut next_agent: Option<String> = None;
        let mut tool_results: Vec<ToolResult> = vec![];

        for tool_call in result.tool_calls {

            match tool_call.type_.as_str() {
                "agent" => {
                    next_agent = Some(tool_call.name);
                },
                "function" => {

                    let read_lock = self.tool_registry.read();
                    let actual_tool = read_lock.unwrap().get_tool(tool_call.name.as_str());
                    match actual_tool {
                        None => {
                            tool_results.push(ToolResult {
                                id: tool_call.id.clone(),
                                code: tool_call.name.clone(),
                                value: json!({ "error": "Tool not found in tool registry."})
                            });
                        }
                        Some(tool) => {
                            let tool_call_result = tool.execute(Some(tool_call.id.clone()), &tool_call.input.unwrap_or(json!({}))).await;
                            match tool_call_result {
                                Ok(tool_call_value) => {
                                    println!("Tool response: {:?}", tool_call_value);
                                    tool_results.push(ToolResult {
                                        id: tool_call_value.id.clone(),
                                        code: tool_call_value.code.clone(),
                                        value: tool_call_value.value,
                                    });
                                }
                                Err(e) => {
                                    tool_results.push(ToolResult {
                                        id: tool_call.id.clone(),
                                        code: tool_call.name.clone(),
                                        value: json!({ "error": "Error calling tool."})
                                    });
                                }
                            }
                        }
                    };
                },
                _ => {}
            }
        }

        SwarmToolResults {
            next_agent,
            tool_results,
        }
    }

    async fn append_tool_call_result_to_memory(&self, tool_results: SwarmToolResults) {

        let read_lock = self.tool_registry.read().unwrap();

        for tool_result in tool_results.tool_results {

            let actual_tool = read_lock.get_tool(tool_result.code.as_str());

            if let Some(tool) = actual_tool {
                let prompt_message = PromptMessage::from_tool_result(
                    &tool_result, tool.as_ref()
                );
                self.memory.record_prompt_messages(&self.memory_context, &vec![prompt_message]).await.unwrap();
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use std::fs;
    use serde_json::{json, Value};
    use testcontainers::core::{IntoContainerPort, Mount, WaitFor};
    use testcontainers::{GenericImage, ImageExt};
    use testcontainers::runners::AsyncRunner;
    use agentsmith_agent::agent::agent::AgentConfig;
    use agentsmith_agent::agent::agent_factory::AgentFactory;
    use agentsmith_common::config::config::read_config;
    use agentsmith_agent::agent::agent_tool::AgentTool;
    use agentsmith_agent::llm::anthropic_llm::ToolChoice::Tool;
    use agentsmith_agent::llm::llm::{LLMConfiguration, LLMCredentials};
    use agentsmith_agent::llm::llm_factory::LLMFactory;
    use agentsmith_agent::llm::prompt::{Prompt, PromptMessage, UserContent};
    use agentsmith_agent::memory::memory::MemoryConfiguration;
    use agentsmith_agent::tools::agent_tool::CallAgentTool;
    use agentsmith_agent::tools::registry::{SafeToolRegistry, ToolRegistry};
    use agentsmith_agent::tools::tool::{Tool as ActualTool, ToolType};
    use agentsmith_agent::tools::web_tool::SimpleJsonWebClientTool;
    use super::*;


    // #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[tokio::test]
    async fn test_call_llm() {

        // tracing_subscriber::fmt::init();

        let current_dir = std::env::current_dir().unwrap();

        let config = read_config("./config.json").unwrap();

        let groq_config = config.clone().config.gateways.registry
            .get("groq_gateway")
            .unwrap()
            .clone();

        let llm_config = LLMConfiguration {
            provider: "groq".to_string(),
            base_url: None,
            model: "llama3-70b-8192".to_string(),
            temperature: None,
            credentials: LLMCredentials {
                api_key: groq_config.api_key
            },
            version: None,
            top_p: None,
            seed: None,
            max_tokens: Some(500),
            stream: Some(false),
        };

        let agent_config = AgentConfig {
            id: "candidate_agent".to_string(),
            name: "Unit Test".to_string(),
            description: "You are a world renowned weather reporter.".to_string(),
            r#type: "simple".to_string(),
            system_prompt: Some("You are a world renowned weather reporter.".to_string()),
            llm: llm_config,
            memory: MemoryConfiguration { r#type: "messages".to_string(), },
            toolbox: vec![AgentTool { code: "get-weather".to_string(), r#type: "function".to_string() }],
        };

        let tool_registry: SafeToolRegistry = ToolRegistry::new();

        let get_weather_tool = SimpleJsonWebClientTool::new(
            "get-weather".to_string(),
            "Get the weather for the unit test".to_string(),
            "http://localhost:1080/agentsmith-agent/unittest/tools/web-tool/1".to_string(),
            "get".to_string(),
            HashMap::new(),
            1000,
            1000,
            1000,
            json!({
          "type": "object",
          "properties": {
            "unit": {
              "type": "string",
              "description": "Degrees celsius or fahrenheit."
            },
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA."
            }
          },
          "required": ["location"]
        }),
            json!({
          "type": "object",
          "properties": {
            "message": {
              "type": "string",
              "description": "General message"
            },
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "temp": {
              "type": "object",
              "properties": {
                    "unit": {
                                "type": "string",
                                "description": "Unit of measure"
                            },
                    "amount": {
                                "type": "number",
                                "description": "numeric representation"
                            }
              }
            }
          },
          "required": ["message"]
        }),
            None,
        );


        tool_registry.write().unwrap().register("get-weather".to_string(), ActualTool::SimpleJsonWebClientTool(get_weather_tool));

        let message = PromptMessage::User { role: "user".to_string(), content: vec![UserContent::Text { type_: "text".to_string(), text: "What is the weather like in Boston today?".to_string() }], name: None };

        let factory = AgentFactory::new(config.clone());

        let agent = &factory.instance(agent_config).await.unwrap();

        let mut swarm = Swarm::new(
            Arc::new(vec![agent.clone()]),
            &tool_registry,
            5,
        );

        let result = swarm.run(message).await.unwrap();

        println!("all messages: {:?}", result.memory.retrieve_past_messages(&swarm.memory_context).await);
    }
}