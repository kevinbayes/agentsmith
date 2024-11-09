use agentsmith_common::config::config::Config;
use agentsmith_common::error::error::{SystemError, SystemResult};
use crate::llm::prompt::PromptMessage;
use crate::memory::general::General;
use crate::memory::messages::Messages;

#[derive(Clone)]
pub enum Memory {
    MESSAGES(Messages),
    GENERAL(General),
}

#[derive(Clone, Debug)]
pub struct MemoryConfiguration {
    pub r#type: String,
}

pub struct MemoryFactory {
    pub config: Config,
}

impl MemoryFactory {
    pub fn new(config: Config) -> Self {
        Self {
            config
        }
    }

    pub async fn instance(&self, memory_config: MemoryConfiguration) -> SystemResult<Memory> {
        match memory_config.r#type.as_ref() {
            "messages" => Ok(Memory::MESSAGES(Messages::new())),
            _ => panic!(),
        }
    }
}

impl RetrieveMemory for Memory {
    async fn retrieve_memory_chunks(&self, collection: &str, query: &str) -> SystemResult<Vec<String>> {
        match self {
            Memory::MESSAGES(memory) => memory.retrieve_memory_chunks(collection, query).await,
            Memory::GENERAL(memory) => memory.retrieve_memory_chunks(collection, query).await,
        }
    }

    async fn retrieve_past_messages(&self) -> SystemResult<Vec<PromptMessage>> {
        match self {
            Memory::MESSAGES(memory) => memory.retrieve_past_messages().await,
            Memory::GENERAL(memory) => memory.retrieve_past_messages().await,
        }
    }
}

impl RecordMemory for Memory {
    async fn record_memory_chunk(&self, collection: &str, chunk: &str) -> SystemResult<bool> {
        match self {
            Memory::MESSAGES(memory) => memory.record_memory_chunk(collection, chunk).await,
            Memory::GENERAL(memory) => memory.record_memory_chunk(collection, chunk).await,
        }
    }

    async fn record_prompt_messages<'a>(&'a self, messages: &'a Vec<PromptMessage>) -> SystemResult<bool> {
        match self {
            Memory::MESSAGES(memory) => memory.record_prompt_messages(messages).await,
            Memory::GENERAL(memory) => memory.record_prompt_messages(messages).await,
        }
    }
}

pub trait InitialiseMemory {
    async fn initialise_collection(&self, collection: &str) -> SystemResult<()>;
}

pub trait RecordMemory {
    async fn record_memory_chunk(&self, collection: &str, chunk: &str) -> SystemResult<bool>;
    async fn record_prompt_message(&self, message: &PromptMessage) -> SystemResult<bool> {
        self.record_prompt_messages(&vec![message.clone()]).await
    }
    async fn record_prompt_messages<'a>(&'a self, messages: &'a Vec<PromptMessage>) -> SystemResult<bool>;
}

pub trait RetrieveMemory {
    async fn retrieve_memory_chunks(&self, collection: &str, query: &str) -> SystemResult<Vec<String>>;
    async fn retrieve_past_messages(&self) -> SystemResult<Vec<PromptMessage>>;
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