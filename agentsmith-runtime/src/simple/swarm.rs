use agentsmith_agent::agent::agent::Agent;
use agentsmith_agent::memory::memory::Memory;
use agentsmith_agent::memory::messages::Messages;
use std::sync::Arc;

pub struct Swarm {
    pub memory: Memory,
    pub agents: Arc<Vec<Agent>>,
    pub initial_agent: String,
    pub active_agent: String,
    pub max_turns: u16,
    pub turn: u16,
}

pub struct SwarmResult {
    pub memory: Memory,
}

impl Swarm {

    pub fn new(agents: Arc<Vec<Agent>>, max_turns: u16, ) -> Self {

        if agents.is_empty() {
            panic!("No agents configured, must have at least one!");
        }

        let initial_agent = agents.first().unwrap().clone().id().to_string();

        Self {
            memory: Memory::MESSAGES(Messages::new()),
            agents: agents.clone(),
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


    pub fn run(&'static mut self) -> SwarmResult {

        while self.turn < self.max_turns {

        }

        let memory = self.memory.clone();

        SwarmResult { memory }
    }
}

