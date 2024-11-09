use std::sync::{Arc, Mutex, RwLock};
use qdrant_client::config::QdrantConfig;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{CreateCollection, CreateCollectionBuilder, Distance, PointStruct, ScalarQuantizationBuilder, VectorParams, VectorParamsBuilder, VectorsConfig};
use qdrant_client::qdrant::qdrant_client::QdrantClient;
use serde::{Deserialize, Serialize};
use agentsmith_common::config::config::Config;
use agentsmith_common::error::error::{SystemError, SystemResult};
use crate::llm::prompt::PromptMessage;
use crate::memory::memory::{InitialiseMemory, RecordMemory, RetrieveMemory};

#[derive(Clone)]
pub struct Messages {
    pub message_log: Arc<RwLock<Vec<PromptMessage>>>,
}

impl Messages {
    pub fn new() -> Self {
        Self {
            message_log: Arc::new(RwLock::new(vec![])),
        }
    }
}

impl RecordMemory for Messages {

    async fn record_memory_chunk(&self, collection: &str, chunk: &str) -> SystemResult<bool> {
        Ok(true)
    }

    async fn record_prompt_messages<'a>(&'a self, messages: &'a Vec<PromptMessage>) -> SystemResult<bool> {
        if messages.is_empty() {
            Ok(false)
        } else {
            let mut result = self.message_log.write().unwrap();
            result.extend(messages.iter().cloned());
            Ok(true)
        }
    }
}

impl RetrieveMemory for Messages {

    async fn retrieve_past_messages(&self) -> SystemResult<Vec<PromptMessage>> {
        Ok(self.message_log.read().unwrap().iter().cloned().collect())
    }

    async fn retrieve_memory_chunks(&self, collection: &str, query: &str) -> SystemResult<Vec<String>> {
        Ok(vec![])
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
    use crate::llm::prompt::UserContent;
    use super::*;


    // #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[tokio::test]
    async fn test_record_prompt_message() {

        tracing_subscriber::fmt::init();

        let client = Qdrant::from_url("http://localhost:6333").build().unwrap();

        let memory = Messages::new();

        let current_storage = memory.retrieve_past_messages().await;

        assert_eq!(current_storage.unwrap().len(), 0);

        let result_true = memory.record_prompt_messages(&vec![PromptMessage::User {
            role: "user".to_string(),
            content: vec![UserContent::Text {
                type_: "text".to_string(),
                text: "Hello world!".to_string(),
            }],
            name: None,
        }]).await.unwrap();

        assert!(result_true);

        let result_false = memory.record_prompt_messages(&vec![]).await.unwrap();

        assert!(!result_false);

        let current_storage = memory.retrieve_past_messages().await.unwrap();

        assert_eq!(current_storage.len(), 1);
    }
}