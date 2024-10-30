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


#[cfg(test)]
mod tests {
    use std::fs;
    use serde_json::{json, Value};
    use testcontainers::core::{IntoContainerPort, Mount, WaitFor};
    use testcontainers::{GenericImage, ImageExt};
    use testcontainers::runners::AsyncRunner;
    use agentsmith_common::config::config::read_config;
    use crate::llm::llm_factory::LLMFactory;
    use super::*;


    // #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[tokio::test]
    async fn test_call_llm() {

        tracing_subscriber::fmt::init();

        let current_dir = std::env::current_dir().unwrap();

        let config = read_config("./secret-config.json").unwrap();

        let factory = LLMFactory::new(config.clone());



    }
}