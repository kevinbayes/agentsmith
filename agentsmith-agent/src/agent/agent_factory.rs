use agentsmith_common::config::config::Config;
use agentsmith_common::error::error::{SystemError, SystemResult};
use crate::agent::agent::{Agent, AgentConfig};
use crate::agent::simple_agent::SimpleAgent;
use crate::llm::llm_factory::LLMFactory;
use crate::memory::memory::MemoryFactory;

struct AgentFactory {
    config: Config,
    llm_factory: LLMFactory,
    memory_factory: MemoryFactory
}

impl AgentFactory {

    pub fn new(config: Config,) ->  Self {

        Self {
            config: config.clone(),
            llm_factory: LLMFactory::new(config.clone()),
            memory_factory: MemoryFactory::new(config.clone()),
        }
    }

    pub async fn instance(&self, agent_config: AgentConfig) -> SystemResult<Agent> {

        match agent_config.r#type.as_str() {
            "simple" => {

                let memory = self.memory_factory
                    .instance(agent_config.memory.clone())
                    .await?;

                let llm = self.llm_factory
                    .instance(agent_config.llm.clone())?;

                Ok(Agent::SimpleAgent(SimpleAgent::new(&agent_config, llm, memory)?))
            },
            _ => Err(SystemError::AgentFactoryError { id: 0, code: 1 })
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
    use agentsmith_common::config::config::read_config;
    use crate::agent::agent_tool::AgentTool;
    use crate::llm::anthropic_llm::ToolChoice::Tool;
    use crate::llm::llm::{LLMConfiguration, LLMCredentials};
    use crate::llm::llm_factory::LLMFactory;
    use crate::llm::prompt::{Prompt, PromptMessage, UserContent};
    use crate::memory::memory::MemoryConfiguration;
    use crate::tools::agent_tool::CallAgentTool;
    use crate::tools::registry::{SafeToolRegistry, ToolRegistry};
    use crate::tools::tool::{Tool as ActualTool, ToolType};
    use super::*;


    // #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[tokio::test]
    async fn test_call_llm() {

        tracing_subscriber::fmt::init();

        let current_dir = std::env::current_dir().unwrap();

        let config = read_config("./secret-config.json").unwrap();

        let cerebras_config = config.clone().config.gateways.registry
            .get("cerebras_gateway")
            .unwrap()
            .clone();

        let llm_config = LLMConfiguration {
            provider: "cerebras".to_string(),
            base_url: None,
            model: "llama3.1-8b".to_string(),
            temperature: None,
            credentials: LLMCredentials {
                api_key: cerebras_config.api_key
            },
            version: None,
            top_p: None,
            seed: None,
            max_tokens: Some(200),
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
        tool_registry.write().unwrap().register("get-weather".to_string(), ActualTool::CallAgentTool(CallAgentTool {
            r#type: ToolType::Agent,
            code: "get-weather".to_string(),
            description: "Get the weather for a given location in celcius or fahrenheit".to_string(),
            input_schema: json!({
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }),
        }));

        let message = vec![
            PromptMessage::System { role: "system".to_string(), content: "You are a world renowned weather reporter.".to_string(), name: None },
            PromptMessage::User { role: "user".to_string(), content: vec![UserContent::Text { type_: "text".to_string(), text: "What's the weather like in Boston today?".to_string() }], name: None },
        ];

        let factory = AgentFactory::new(config.clone());

        let agent = &factory.instance(agent_config).await.unwrap();

        let prompt = Prompt::new_message_for_agent(agent, message, &tool_registry,);

        let result = agent.chat_completion(&prompt).await;

        assert!(result.is_ok());

        println!("{:?}", result.unwrap());

    }
}