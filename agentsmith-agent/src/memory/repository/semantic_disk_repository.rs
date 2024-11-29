use crate::memory::memory::{InitialiseMemory, MemoryBlock};
use agentsmith_common::error::error::SystemResult;
use crate::memory::repository::semantic_repository::{SemanticCommand, SemanticMemoryConfiguration, SemanticQuery, SemanticRepository};
use crate::memory::repository::working_repository::WorkingMemoryConfiguration;

#[derive(Clone)]
pub struct SemanticDiskRepository {
    pub agent_id: String,
    pub config: SemanticMemoryConfiguration,
}

impl SemanticDiskRepository {

    pub async fn new(config: &SemanticMemoryConfiguration) -> SystemResult<Self> {

        Ok(Self {
            agent_id: config.id.clone(),
            config: config.clone(),
        })
    }
}


impl InitialiseMemory for SemanticDiskRepository {
    async fn initialise(&self) -> SystemResult<()> {
        todo!()
    }
}

impl SemanticCommand for SemanticDiskRepository {
    async fn record_memory_chunk(&self, chunk: &str) -> SystemResult<bool> {
        todo!()
    }
}

impl SemanticQuery for SemanticDiskRepository {
    async fn retrieve_memory_chunks(&self, query: &str) -> SystemResult<Vec<MemoryBlock>> {
        todo!()
    }
}