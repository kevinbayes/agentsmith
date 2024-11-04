use std::collections::HashMap;
use agentsmith_agent::agent::agent::Agent;
use agentsmith_agent::memory::memory::{Memory, RecordMemory, RetrieveMemory};
use agentsmith_agent::memory::messages::Messages;
use std::sync::Arc;
use agentsmith_agent::llm::prompt::{Prompt, Tool};
use agentsmith_common::error::error::SystemResult;

pub struct Swarm {
    pub memory: Memory,
    pub agents: Arc<Vec<Agent>>,
    pub tool_registry: HashMap<String, Tool>,
    pub initial_agent: String,
    pub active_agent: String,
    pub max_turns: u16,
    pub turn: u16,
}

pub struct SwarmResult {
    pub memory: Memory,
}

impl Swarm {

    pub fn new(agents: Arc<Vec<Agent>>, tool_registry: &HashMap<String, Tool>, max_turns: u16, ) -> Self {

        if agents.is_empty() {
            panic!("No agents configured, must have at least one!");
        }

        let initial_agent = agents.first().unwrap().clone().id().to_string();

        Self {
            memory: Memory::MESSAGES(Messages::new()),
            agents: agents.clone(),
            tool_registry: tool_registry.clone(),
            initial_agent: initial_agent.clone(),
            active_agent: initial_agent.clone(),
            max_turns,
            turn: 0,
        }
    }

    pub fn reset(&mut self) {

        self.memory = Memory::MESSAGES(Messages::new());
        self.active_agent = self.initial_agent.clone();
        self.turn = 0;
    }


    pub async fn run(&'static mut self) -> SystemResult<SwarmResult> {

        while self.turn < self.max_turns {

            let agent = self.agents.iter()
                .find(|item| item.id() == self.active_agent.clone())
                .unwrap();

            let messages = self.memory.retrieve_past_messages()
                .await?
                .clone();

            let prompt = Prompt::new_message_for_agent(agent, messages, &self.tool_registry,);

            let result = agent.chat_completion(&prompt).await?;

            let call_tools = !result.tool_calls.is_empty();



            self.memory.record_prompt_messages()

        }

        let memory = self.memory.clone();

        Ok(SwarmResult { memory })
    }
}

