use serde::{Deserialize, Serialize};
use agentsmith_common::config::arango::ArangoConfig;
use agentsmith_common::config::config::{Config, QdrantConfig};
use agentsmith_common::error::error::SystemResult;
use crate::llm::prompt::PromptMessage;
use crate::memory::memory::{InitialiseMemory, MemoryContext};
use crate::memory::repository::working_arango_repository::WorkingMemoryArangoRepository;
use crate::memory::repository::working_disk_repository::WorkingMemoryDiskRepository;

#[derive(Clone)]
pub enum WorkingMemoryRepository {
    Disk(WorkingMemoryDiskRepository),
    Arango(WorkingMemoryArangoRepository),
}

#[derive(Clone, Serialize, Deserialize)]
pub struct WorkingMemoryConfiguration {
    pub id: String,
    pub(crate) r#type: String,
    pub disk: Option<WorkingMemoryDiskRepositoryConfiguration>,
    pub arango: Option<WorkingMemoryArangoRepositoryConfiguration>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct WorkingMemoryDiskRepositoryConfiguration {
    pub path: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct WorkingMemoryArangoRepositoryConfiguration {
    pub connection: ArangoConfig,
    pub index: QdrantConfig,
}

pub struct WorkingMemoryRepositoryFactory {
    pub config: Config,
}

impl WorkingMemoryRepositoryFactory {

    pub fn new(config: &Config) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub async fn instance(&self, working_memory_config: &WorkingMemoryConfiguration) -> SystemResult<WorkingMemoryRepository> {
        match working_memory_config.r#type.as_ref() {
            "disk" => {
                let disk = WorkingMemoryDiskRepository::new(working_memory_config)
                    .await?;
                Ok(WorkingMemoryRepository::Disk(disk))
            },
            "arango" => {
                let arango = WorkingMemoryArangoRepository::new(working_memory_config)
                    .await?;
                Ok(WorkingMemoryRepository::Arango(arango))
            },
            _ => panic!(),
        }
    }
}

impl InitialiseMemory for WorkingMemoryRepository {
    async fn initialise(&self) -> SystemResult<()> {
        match self {
            WorkingMemoryRepository::Disk(disk) => {
                disk.initialise().await
            }
            WorkingMemoryRepository::Arango(arango) => {
                arango.initialise().await
            }
        }
    }
}

pub trait WorkingMemoryCommand {
    async fn record_prompt_message(&self, context: &MemoryContext, message: &PromptMessage) -> SystemResult<bool> {
        self.record_prompt_messages(context, &vec![message.clone()]).await
    }
    async fn record_prompt_messages<'a>(&'a self, context: &MemoryContext, messages: &'a Vec<PromptMessage>) -> SystemResult<bool>;
}

impl WorkingMemoryCommand for WorkingMemoryRepository {
    async fn record_prompt_messages<'a>(&'a self, context: &MemoryContext, messages: &'a Vec<PromptMessage>) -> SystemResult<bool> {
        todo!()
    }
}

pub trait WorkingMemoryQuery {
    async fn retrieve_past_messages(&self, context: &MemoryContext) -> SystemResult<Vec<PromptMessage>> {
        self.retrieve_past_n_messages(context, -1).await
    }
    async fn retrieve_past_n_messages(&self, context: &MemoryContext, last_n: i32) -> SystemResult<Vec<PromptMessage>>;
}

impl WorkingMemoryQuery for WorkingMemoryRepository {
    async fn retrieve_past_n_messages(&self, context: &MemoryContext, last_n: i32) -> SystemResult<Vec<PromptMessage>> {
        todo!()
    }
}