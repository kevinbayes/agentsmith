use serde::{Deserialize, Serialize};
use agentsmith_common::config::config::Config;
use agentsmith_common::error::error::{SystemError, SystemResult};
use crate::llm::prompt::PromptMessage;
use crate::memory::general::{General, GeneralMemoryConfiguration};
use crate::memory::messages::Messages;

#[derive(Clone)]
pub enum Memory {
    MESSAGES(Messages),
    GENERAL(General),
}

#[derive(Clone, Debug, Deserialize)]
pub struct MemoryBlock {
    pub address: String,
    pub string: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryConfiguration {
    pub r#type: String,
    pub general: Option<GeneralMemoryConfiguration>,
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

    pub async fn instance(&self, memory_config: &MemoryConfiguration) -> SystemResult<Memory> {

        match memory_config.r#type.as_ref() {
            "messages" => Ok(Memory::MESSAGES(Messages::new())),
            "general" => Ok(Memory::GENERAL(General::new(&self.config, memory_config).await)),
            _ => panic!(),
        }
    }
}

impl RetrieveMemory for Memory {
    async fn retrieve_memory_chunks(&self, query: &str) -> SystemResult<Vec<MemoryBlock>> {
        match self {
            Memory::MESSAGES(memory) => memory.retrieve_memory_chunks(query).await,
            Memory::GENERAL(memory) => memory.retrieve_memory_chunks(query).await,
        }
    }

    async fn retrieve_past_n_messages(&self, context: &MemoryContext, last_n: i32) -> SystemResult<Vec<PromptMessage>> {
        match self {
            Memory::MESSAGES(memory) => memory.retrieve_past_n_messages(context, last_n).await,
            Memory::GENERAL(memory) => memory.retrieve_past_n_messages(context, last_n).await,
        }
    }
}

impl RecordMemory for Memory {
    async fn record_memory_chunk(&self, chunk: &str) -> SystemResult<bool> {
        match self {
            Memory::MESSAGES(memory) => memory.record_memory_chunk(chunk).await,
            Memory::GENERAL(memory) => memory.record_memory_chunk(chunk).await,
        }
    }

    async fn record_prompt_messages<'a>(&'a self, context: &MemoryContext, messages: &'a Vec<PromptMessage>) -> SystemResult<bool> {
        match self {
            Memory::MESSAGES(memory) => memory.record_prompt_messages(context, messages).await,
            Memory::GENERAL(memory) => memory.record_prompt_messages(context, messages).await,
        }
    }
}

pub struct MemoryContext {
    pub interaction_id: String,
}

pub trait InitialiseMemory {
    async fn initialise(&self) -> SystemResult<()>;
}

pub trait RecordMemory {
    async fn record_memory_chunk(&self, chunk: &str) -> SystemResult<bool>;
    async fn record_prompt_message(&self, context: &MemoryContext, message: &PromptMessage) -> SystemResult<bool> {
        self.record_prompt_messages(context, &vec![message.clone()]).await
    }
    async fn record_prompt_messages<'a>(&'a self, context: &MemoryContext, messages: &'a Vec<PromptMessage>) -> SystemResult<bool>;
}

pub trait RetrieveMemory {
    async fn retrieve_memory_chunks(&self, query: &str) -> SystemResult<Vec<MemoryBlock>>;
    async fn retrieve_past_messages(&self, context: &MemoryContext) -> SystemResult<Vec<PromptMessage>> {
        self.retrieve_past_n_messages(context, -1).await
    }
    async fn retrieve_past_n_messages(&self, context: &MemoryContext, last_n: i32) -> SystemResult<Vec<PromptMessage>>;
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