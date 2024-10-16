use agentsmith_common::config::config::Config;

struct AgentFactory {
    config: Config
}

impl AgentFactory {

    pub fn new(config: Config) ->  Self {
        Self {
            config: config.clone(),
        }
    }
}