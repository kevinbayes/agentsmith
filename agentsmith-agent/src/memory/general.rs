use crate::llm::prompt::PromptMessage;
use crate::memory::memory::{InitialiseMemory, MemoryBlock, MemoryContext, RecordMemory, RetrieveMemory};
use crate::memory::repository::semantic_repository::{SemanticCommand, SemanticMemoryConfiguration, SemanticQuery, SemanticRepository, SemanticRepositoryFactory};
use crate::memory::repository::working_repository::{WorkingMemoryCommand, WorkingMemoryConfiguration, WorkingMemoryQuery, WorkingMemoryRepository, WorkingMemoryRepositoryFactory};
use agentsmith_common::config::config::Config;
use agentsmith_common::error::error::SystemResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone)]
pub struct General {
    pub working: Arc<WorkingMemoryRepository>,
    pub semantic: Arc<SemanticRepository>,
}

impl General {

    pub async fn new(config: &Config,
                     working_memory_configuration: &WorkingMemoryConfiguration,
                     semantic_repository_configuration: &SemanticMemoryConfiguration) -> Self {

        let working_factory = WorkingMemoryRepositoryFactory::new(config);
        let semantic_factory = SemanticRepositoryFactory::new(config);

        let working = working_factory.instance(
            working_memory_configuration
        )
            .await
            .unwrap();

        let semantic = semantic_factory.instance(
            semantic_repository_configuration
        )
            .await
            .unwrap();

        Self {
            working: Arc::new(working),
            semantic: Arc::new(semantic),
        }
    }
}

impl InitialiseMemory for General {

    async fn initialise(&self) -> SystemResult<()> {
        self.working.initialise().await?;
        self.semantic.initialise().await
    }
}

impl RecordMemory for General {

    async fn record_memory_chunk(&self, chunk: &str) -> SystemResult<bool> {
        self.semantic.record_memory_chunk(chunk).await
    }

    async fn record_prompt_messages<'a>(&'a self, context: &MemoryContext, messages: &'a Vec<PromptMessage>) -> SystemResult<bool> {
        self.working.record_prompt_messages(context, messages).await
    }
}

impl RetrieveMemory for General {

    async fn retrieve_past_n_messages(&self, context: &MemoryContext, last_n: i32) -> SystemResult<Vec<PromptMessage>> {
        self.working.retrieve_past_n_messages(context, last_n).await
    }

    async fn retrieve_memory_chunks(&self, query: &str) -> SystemResult<Vec<MemoryBlock>> {
        self.semantic.retrieve_memory_chunks(query).await
    }
}



#[cfg(test)]
mod tests {
    // #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[tokio::test]
    async fn test_record_prompt_message() {

        tracing_subscriber::fmt::init();

        // let client = Qdrant::from_url("http://localhost:6333").build().unwrap();
        //
        // let memory = General {
        //     message_log: Arc::new(RwLock::new(vec![])),
        //     memory: Arc::new(client),
        // };
        //
        // memory.initialise(&String::from("test")).await.unwrap();
        //
        // let current_storage = memory.retrieve_past_messages().await;
        //
        // assert_eq!(current_storage.unwrap().len(), 0);
        //
        // let result_true = memory.record_prompt_messages(&vec![PromptMessage::User {
        //     role: "user".to_string(),
        //     content: vec![UserContent::Text {
        //         type_: "text".to_string(),
        //         text: "Hello world!".to_string(),
        //     }],
        //     name: None,
        // }]).await.unwrap();
        //
        // assert!(result_true);
        //
        // let result_false = memory.record_prompt_messages(&vec![]).await.unwrap();
        //
        // assert!(!result_false);
        //
        // let current_storage = memory.retrieve_past_messages().await.unwrap();
        //
        // assert_eq!(current_storage.len(), 1);
    }
}