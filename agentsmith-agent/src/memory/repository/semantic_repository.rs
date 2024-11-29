use crate::memory::memory::{InitialiseMemory, MemoryBlock};
use crate::memory::repository::semantic_arango_repository::SemanticArangoRepository;
use crate::memory::repository::semantic_disk_repository::SemanticDiskRepository;
use agentsmith_common::config::arango::ArangoConfig;
use agentsmith_common::config::config::{Config, QdrantConfig};
use agentsmith_common::error::error::SystemResult;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub enum SemanticRepository {
    Disk(SemanticDiskRepository),
    Arango(SemanticArangoRepository),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticMemoryConfiguration {
    pub id: String,
    pub r#type: String,
    pub disk: Option<SemanticDiskRepositoryConfiguration>,
    pub arango: Option<SemanticArangoRepositoryConfiguration>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticDiskRepositoryConfiguration {
    pub path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticArangoRepositoryConfiguration {
    pub connection: ArangoConfig,
    pub index: QdrantConfig,
}


pub struct SemanticRepositoryFactory {
    pub config: Config,
}

impl SemanticRepositoryFactory {

    pub fn new(config: &Config) -> Self {
        Self {
            config: config.clone()
        }
    }

    pub async fn instance(&self, semantic_memory_configuration: &SemanticMemoryConfiguration) -> SystemResult<SemanticRepository> {
        let memory_type: &str = semantic_memory_configuration.r#type.as_str();
        match memory_type {
            "disk" => {
                Ok(SemanticRepository::Disk(SemanticDiskRepository::new(semantic_memory_configuration).await?))
            }
            "arango" => {
                Ok(SemanticRepository::Arango(SemanticArangoRepository::new(semantic_memory_configuration).await?))
            }
            _ => panic!(),
        }
    }
}

pub struct Query {
    q: String,
}

impl InitialiseMemory for SemanticRepository {
    async fn initialise(&self) -> SystemResult<()> {
        match self {
            SemanticRepository::Disk(disk) => {
                disk.initialise().await
            }
            SemanticRepository::Arango(arango) => {
                arango.initialise().await
            }
        }
    }
}

pub trait SemanticCommand {
    async fn record_memory_chunk(&self, chunk: &str) -> SystemResult<bool>;
}

impl SemanticCommand for SemanticRepository {
    async fn record_memory_chunk(&self, chunk: &str) -> SystemResult<bool> {
        todo!()
    }
}

pub trait SemanticQuery {
    async fn retrieve_memory_chunks(&self, query: &str) -> SystemResult<Vec<MemoryBlock>>;
}

impl SemanticQuery for SemanticRepository {
    async fn retrieve_memory_chunks(&self, query: &str) -> SystemResult<Vec<MemoryBlock>> {
        todo!()
    }
}