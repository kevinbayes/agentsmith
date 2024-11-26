use crate::llm::prompt::PromptMessage;
use crate::memory::memory::{InitialiseMemory, MemoryContext};
use crate::memory::repository::working_repository::{
    WorkingMemoryCommand, WorkingMemoryConfiguration, WorkingMemoryQuery,
};
use agentsmith_common::error::error::{SystemError, SystemResult};
use arangors::client::reqwest::ReqwestClient;
use arangors::document::options::InsertOptions;
use arangors::{AqlQuery, Collection, Connection, Database};
use chrono::{DateTime, Utc};
use log::debug;
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use arangors::transaction::{Status, TransactionCollections, TransactionSettings};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
struct WorkingMemoryItem {
    #[serde(rename = "_key")]
    id: String,
    agent_id: String,
    message: PromptMessage,
    created_on: DateTime<Utc>,
    created_by: String,
    modified_on: DateTime<Utc>,
    modified_by: String,
}

#[derive(Clone)]
pub struct WorkingMemoryArangoRepository {
    pub agent_id: String,
    pub repository: Arc<Connection>,
    pub index: Arc<Qdrant>,
}

impl WorkingMemoryArangoRepository {
    const DATABASE_NAME: &'static str = "agentsmith";
    const COLLECTION_NAME: &'static str = "working-memory";
    const INDEX_NAME: &'static str = "working-memory";

    pub async fn new(config: &WorkingMemoryConfiguration) -> SystemResult<Self> {
        let arango_config = config.arango.clone().unwrap();

        let qdrant_config = arango_config.index.clone();
        let index = Qdrant::from_url(qdrant_config.host.as_str())
            .build()
            .unwrap();

        let repository_config = arango_config.connection.clone();
        let arango_url = format!(
            "{}://{}:{}",
            repository_config.protocol, repository_config.host, repository_config.port
        );
        let repository = Connection::establish_jwt(
            arango_url.as_str(),
            repository_config.user.as_str(),
            repository_config.pass.as_str(),
        )
        .await
        .unwrap();

        Ok(Self {
            agent_id: config.id.clone(),
            index: Arc::new(index),
            repository: Arc::new(repository),
        })
    }

    pub async fn database(&self) -> SystemResult<Database<ReqwestClient>> {
        Ok(self
            .repository
            .db(Self::DATABASE_NAME)
            .await
            .map_err(|e| SystemError::MemoryError { id: 0, code: 1 })?)
    }

    pub async fn collection(&self) -> SystemResult<Collection<ReqwestClient>> {
        let db = self.database().await?;

        match db.collection(Self::COLLECTION_NAME).await {
            Ok(collection) => Ok(collection),
            Err(_) => Ok(db
                .create_collection(Self::COLLECTION_NAME)
                .await
                .map_err(|e| SystemError::MemoryError { id: 0, code: 2 })?),
        }
    }
}

impl InitialiseMemory for WorkingMemoryArangoRepository {
    async fn initialise(&self) -> SystemResult<()> {
        let collection = self.collection().await?;
        println!("Collection {:?}", collection);
        Ok(())
    }
}

impl WorkingMemoryCommand for WorkingMemoryArangoRepository {
    async fn record_prompt_messages<'a>(
        &'a self,
        context: &MemoryContext,
        messages: &'a Vec<PromptMessage>,
    ) -> SystemResult<bool> {
        // Ensure the collection exists
        let db = self.database().await?;

        let now = Utc::now();

        let mut transaction_actions = Vec::new();

        // Convert messages to working memory items and prepare transaction actions
        for msg in messages {
            let memory_item = WorkingMemoryItem {
                id: Uuid::new_v4().to_string(),
                agent_id: self.agent_id.clone(),
                message: msg.clone(),
                created_on: now,
                created_by: self.agent_id.clone(),
                modified_on: now,
                modified_by: self.agent_id.clone(),
            };

            // Prepare parameters for the action
            let doc = serde_json::json!({
                "id": memory_item.id,
                "agent_id": memory_item.agent_id,
                "message": memory_item.message,
                "created_on": memory_item.created_on,
                "created_by": memory_item.created_by,
                "modified_on": memory_item.modified_on,
                "modified_by": memory_item.modified_by
            });

            transaction_actions.push(doc);
        }

        // Configure transaction options
        let transaction_settings = TransactionSettings::builder()
            .lock_timeout(60000)
            .wait_for_sync(true)
            .collections(
                TransactionCollections::builder()
                    .write(vec![Self::COLLECTION_NAME.clone().to_owned()])
                    .build(),
            )
            .build();

        // Begin transaction
        let mut transaction = db.begin_transaction(transaction_settings)
            .await
            .map_err(|e| {
                println!("Failed to insert documents - {}", e);
                SystemError::MemoryError { id: 3, code: 1000 }
            })?;


        let collection = transaction.collection(Self::COLLECTION_NAME)
            .await
            .map_err(|e| {
                println!("Failed to insert documents - {}", e);
                SystemError::MemoryError { id: 3, code: 1001 }
            })?;

        // Execute each document insert within the transaction
        for doc in transaction_actions {
            collection.create_document(doc, InsertOptions::builder().build())
                .await
                .map_err(|e| {
                    println!("Failed to insert documents - {}", e);
                    SystemError::MemoryError { id: 3, code: 1002 }
                })?;
        }

        // Commit transaction
        let commit_result = transaction.commit()
            .await
            .map_err(|e| {
                println!("Failed to insert documents - {}", e);
                SystemError::MemoryError { id: 3, code: 1003 }
            })?;

        // Check transaction status
        match commit_result {
            Status::Committed => Ok(true),
            _ => {
                println!("Failed to insert documents.");
                Err(SystemError::MemoryError { id: 3, code: 1004 })
            }
        }
    }
}

impl WorkingMemoryQuery for WorkingMemoryArangoRepository {
    async fn retrieve_past_n_messages(&self, context: &MemoryContext, last_n: i32) -> SystemResult<Vec<PromptMessage>> {
        let db = self.database().await?;

        let limit = if last_n < 1 { 2000 } else { last_n };
        println!("Limit is {}", limit);

        // AQL query to get the last n messages ordered by creation time
        let aql = AqlQuery::builder()
            .query("FOR u IN @@collection FILTER u.agent_id==@agent_id LIMIT @limit RETURN u")
            .bind_var("@collection", Self::COLLECTION_NAME)
            .bind_var("agent_id", self.agent_id.clone())
            .bind_var("limit", limit)
            .build();

        let messages: Vec<WorkingMemoryItem> = db
            .aql_query(aql)
            .await
            .map_err(|e| SystemError::MemoryError { id: 0, code: 0 })?;

        Ok(messages.iter().map(|i| i.message.clone()).collect())
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, thread};
    use std::path::Path;
    use std::time::Duration;
    use arangors::Connection;
    use log::Level;
    use testcontainers::*;
    use testcontainers::{
        core::{IntoContainerPort, WaitFor},
        runners::AsyncRunner,
        GenericImage,
    };
    use testcontainers::core::logs::consumer::LogConsumer;
    use testcontainers::core::logs::consumer::logging_consumer::LoggingConsumer;
    use testcontainers::core::logs::LogFrame;
    use futures::{future::BoxFuture, FutureExt};
    use agentsmith_common::config::arango::ArangoConfig;
    use agentsmith_common::config::config::QdrantConfig;
    use crate::llm::prompt::{PromptMessage, UserContent};
    use crate::memory::memory::{InitialiseMemory, MemoryContext};
    use crate::memory::repository::working_arango_repository::WorkingMemoryArangoRepository;
    use crate::memory::repository::working_repository::{WorkingMemoryArangoRepositoryConfiguration, WorkingMemoryCommand, WorkingMemoryConfiguration, WorkingMemoryQuery};

    struct TestFixture {
        id: String,
        container_arango: ContainerAsync<GenericImage>,
        container_index: ContainerAsync<GenericImage>,
    }

    #[derive(Clone)]
    struct LogPrinter{}

    impl LogConsumer for LogPrinter {
        fn accept<'a>(&'a self, record: &'a LogFrame) -> BoxFuture<'a, ()> {

            async move {
                match record {
                    LogFrame::StdOut(bytes) => {
                        println!("container: {:?}", String::from_utf8_lossy(bytes));
                    }
                    LogFrame::StdErr(bytes) => {
                        println!("container: {:?}", String::from_utf8_lossy(bytes));
                    }
                }
            }.boxed()
        }
    }

    impl TestFixture {
        async fn new() -> Self {

            let log_consumer = LogPrinter {};

            let container_arango = GenericImage::new("arangodb", "3.12")
                .with_wait_for(WaitFor::message_on_stdout("Have fun!"))
                .with_mapped_port(18529, 8529.tcp())
                .with_log_consumer(log_consumer.clone())
                .with_env_var("ARANGO_ROOT_PASSWORD", "password")
                .start()
                .await
                .expect("Arango started");

            let container_index = GenericImage::new("qdrant/qdrant", "latest")
                .with_wait_for(WaitFor::message_on_stdout("Qdrant gRPC listening on 6334"))
                .with_mapped_port(16333, 6333.tcp())
                .with_mapped_port(16334, 6334.tcp())
                .with_log_consumer(log_consumer.clone())
                .start()
                .await
                .expect("Qdrant started");

            let repository = Connection::establish_jwt(
                "http://localhost:18529",
                "root",
                "password",
            )
                .await
                .unwrap();

            repository.create_database(WorkingMemoryArangoRepository::DATABASE_NAME).await.unwrap();

            let id = "unittest-1".to_string();

            Self {
                id,
                container_arango,
                container_index,
            }
        }
    }

    impl Drop for TestFixture {
        fn drop(&mut self) {
        }
    }

    #[tokio::test]
    async fn test_working_memory_initialisation() {
        let fixture = TestFixture::new().await;
        // thread::sleep(Duration::from_secs(30));
        println!("Container arango started: {:?}", fixture.container_arango);
        println!("Container index started: {:?}", fixture.container_index);

        let working_memory_config = WorkingMemoryConfiguration {
            id: "agent-ut".to_string(),
            r#type: "arango".to_string(),
            disk: None,
            arango: Some(WorkingMemoryArangoRepositoryConfiguration {
                index: QdrantConfig {
                    host: "http://192.168.1.151:16334".to_string(),
                },
                connection: ArangoConfig {
                    protocol: "http".to_string(),
                    host: "localhost".to_string(),
                    port: "18529".to_string(),
                    user: "root".to_string(),
                    pass: "password".to_string()
                }
            })
        };

        let repository = WorkingMemoryArangoRepository::new(&working_memory_config).await.unwrap();

        let _ = repository.initialise().await;

        let context = MemoryContext {
            interaction_id: "ut".to_string(),
        };

        let prompt = PromptMessage::System {
            name: Some("system".to_string()),
            content: String::from("Test system"),
            role: "system".to_string(),
        };

        let result = repository.record_prompt_message(&context, &prompt).await.unwrap();
        assert!(result);

        let saved_messages = &repository.retrieve_past_messages(&context).await.unwrap();
        println!("Saved messages {:?}", saved_messages);


        let prompt = PromptMessage::User {
            name: Some("kevin".to_string()),
            content: vec![UserContent::Text { type_: "text".to_string(), text: "Hello world!".to_string() }],
            role: "user".to_string(),
        };

        let result = repository.record_prompt_message(&context, &prompt).await.unwrap();
        assert!(result);

        let saved_messages = &repository.retrieve_past_messages(&context).await.unwrap();

        let saved_system_message = saved_messages.get(0).unwrap();
        let saved_user_message = saved_messages.get(1).unwrap();

        assert_eq!(2, saved_messages.len());

        assert_eq!("system".to_string(), saved_system_message.role().clone());
        assert_eq!("user".to_string(), saved_user_message.role().clone());
    }
}
