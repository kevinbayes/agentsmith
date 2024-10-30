use agentsmith_common::error::error::SystemError;
use crate::llm::prompt::PromptMessage;
use crate::memory::general::General;

pub enum Memory {
    GENERAL(General),
}

pub trait InitialiseMemory {
    async fn initialise_collection(&self, collection: &str) -> Result<(), SystemError>;
}

pub trait RecordMemory {
    async fn record_memory_chunk(&self, collection: &str, chunk: &str) -> Result<bool, SystemError>;
    async fn record_prompt_messages<'a>(&'a self, messages: &'a Vec<PromptMessage>) -> bool;
}

pub trait RetrieveMemory {
    async fn retrieve_memory_chunks(&self, collection: &str, query: &str) -> Vec<String>;
    async fn retrieve_past_messages(&self) -> Vec<PromptMessage>;
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


    }
}