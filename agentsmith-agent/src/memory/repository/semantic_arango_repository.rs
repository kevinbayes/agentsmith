use arangors::client::reqwest::ReqwestClient;
use std::sync::Arc;

use crate::memory::memory::{InitialiseMemory, MemoryBlock};
use crate::memory::repository::semantic_repository::{SemanticCommand, SemanticMemoryConfiguration, SemanticQuery};
use crate::memory::repository::working_repository::WorkingMemoryConfiguration;
use agentsmith_common::error::error::{SystemError, SystemResult};
use arangors::{Collection, Connection, Database};
use qdrant_client::Qdrant;

#[derive(Clone)]
pub struct SemanticArangoRepository {
    pub agent_id: String,
    pub repository: Arc<Connection>,
    pub index: Arc<Qdrant>,
}


impl SemanticArangoRepository {

    const DATABASE_NAME: &'static str = "agentsmith";
    const COLLECTION_NAME: &'static str = "semantic-memory";
    const INDEX_NAME: &'static str = "semantic-memory";

    pub async fn new(config: WorkingMemoryConfiguration) -> SystemResult<Self> {

        let arango_config = config.arango.unwrap();

        let qdrant_config = arango_config.index.clone();
        let index = Qdrant::from_url(qdrant_config.host.as_str())
            .build()
            .unwrap();

        let repository_config = arango_config.connection.clone();
        let arango_url = format!("{}://{}:{}", repository_config.protocol, repository_config.host, repository_config.port);
        let repository = Connection::establish_jwt(arango_url.as_str(),
                                                   repository_config.user.as_str(),
                                                   repository_config.pass.as_str())
            .await
            .unwrap();

        Ok(Self {
            agent_id: config.id.clone(),
            index: Arc::new(index),
            repository: Arc::new(repository),
        })
    }

    async fn database(&self) -> SystemResult<Database<ReqwestClient>> {
        Ok(self.repository.db(Self::DATABASE_NAME)
            .await
            .map_err(|e| SystemError::MemoryError { id: 0, code: 1})
            ?)
    }

    async fn collection(&self) -> SystemResult<Collection<ReqwestClient>> {
        let db = self.database().await?;

        match db.collection(Self::COLLECTION_NAME).await {
            Ok(collection) => Ok(collection),
            Err(_) => Ok(db.create_collection(Self::COLLECTION_NAME).await.map_err(|e| SystemError::MemoryError { id: 0, code: 2})?),
        }
    }
}

impl InitialiseMemory for SemanticArangoRepository {
    async fn initialise(&self) -> SystemResult<()> {
        todo!()
    }
}

impl SemanticCommand for SemanticArangoRepository {
    async fn record_memory_chunk(&self, chunk: &str) -> SystemResult<bool> {
        todo!()
    }
}

impl SemanticQuery for SemanticArangoRepository {
    async fn retrieve_memory_chunks(&self, query: &str) -> SystemResult<Vec<MemoryBlock>> {
        todo!()
    }
}