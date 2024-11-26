use crate::memory::memory::{InitialiseMemory, MemoryBlock};
use agentsmith_common::error::error::SystemResult;
use crate::memory::repository::semantic_repository::{SemanticCommand, SemanticQuery, SemanticRepository};

#[derive(Clone)]
pub struct SemanticDiskRepository {
    pub agent_id: String,
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