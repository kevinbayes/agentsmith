use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::{Arc, RwLock};
use arangors::AqlQuery;
use chrono::{DateTime, Utc};
use log::debug;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use agentsmith_common::error::error::{SystemError, SystemResult};
use crate::llm::prompt::PromptMessage;
use crate::memory::general::General;
use crate::memory::memory::{InitialiseMemory, MemoryContext};
use crate::memory::messages::Messages;
use crate::memory::repository::working_arango_repository::WorkingMemoryArangoRepository;
use crate::memory::repository::working_repository::{WorkingMemoryCommand, WorkingMemoryConfiguration, WorkingMemoryQuery};

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
pub struct WorkingMemoryDiskRepository {
    pub agent_id: String,
    pub config: WorkingMemoryConfiguration,
}

impl WorkingMemoryDiskRepository {

    pub async fn new(config: &WorkingMemoryConfiguration) -> SystemResult<Self> {

        Ok(Self {
            agent_id: config.id.clone(),
            config: config.clone(),
        })
    }

    pub fn file_path(&self,context: &MemoryContext) -> SystemResult<String> {

        let disk_config = self.config.disk.as_ref().ok_or(SystemError::MemoryError {
            code: 2009,
            id: 1
        })?;

        Ok(format!("{}/{}/working-{}.db", disk_config.path, self.agent_id, context.interaction_id))
    }

    pub fn open_file(&self, context: &MemoryContext) -> SystemResult<File> {

        let file_path_str = self.file_path(context)?;

        let file_path = Path::new(file_path_str.as_str());

        if !file_path.exists() {
            File::create(file_path)
                .map_err(|e| {
                    println!("Error creating file {}.", e);
                    SystemError::MemoryError { code: 2008, id: 1 }
                })?;
        }

        let file = File::open(file_path).map_err(|e| {
            debug!("Error opening file for reading: {}", e);
            SystemError::MemoryError { code: 2007, id: 1 }
        })?;

        Ok(file)
    }
}


impl WorkingMemoryCommand for WorkingMemoryDiskRepository {

    async fn record_prompt_messages<'a>(&'a self, context: &MemoryContext, messages: &'a Vec<PromptMessage>) -> SystemResult<bool> {

        let file_path = self.file_path(context)?;

        // Read existing content
        let file = self.open_file(context)?;

        let mut reader = BufReader::new(file);
        let mut content = String::new();
        reader.read_to_string(&mut content).map_err(|e| {
            println!("Error reading file: {}", e);
            SystemError::MemoryError { code: 2000, id: 1 }
        })?;

        // Parse existing content or create new array
        let mut memory_items: Vec<WorkingMemoryItem> = if content.is_empty() {
            Vec::new()
        } else {
            serde_json::from_str(&content).map_err(|e| {
                debug!("Error parsing JSON: {}", e);
                SystemError::MemoryError { code: 2001, id: 1 }
            })?
        };

        // Add new messages
        let now = Utc::now();
        for message in messages {
            let item = WorkingMemoryItem {
                id: Uuid::new_v4().to_string(),
                agent_id: self.agent_id.clone(),
                message: message.clone(),
                created_on: now,
                created_by: "system".to_string(),
                modified_on: now,
                modified_by: "system".to_string(),
            };
            memory_items.push(item);
        }

        // Write back to file
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&file_path)
            .map_err(|e| {
                debug!("Error opening file for writing: {}", e);
                SystemError::MemoryError { code: 2002, id: 1 }
            })?;

        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &memory_items).map_err(|e| {
            debug!("Error writing JSON: {}", e);
            SystemError::MemoryError { code: 2003, id: 1 }
        })?;

        Ok(true)
    }
}

impl WorkingMemoryQuery for WorkingMemoryDiskRepository {

    async fn retrieve_past_n_messages(&self, context: &MemoryContext, last_n: i32) -> SystemResult<Vec<PromptMessage>> {

        // Read file content
        let file = self.open_file(context)?;

        let mut reader = BufReader::new(file);
        let mut content = String::new();
        reader.read_to_string(&mut content).map_err(|e| {
            debug!("Error reading file: {}", e);
            SystemError::MemoryError { code: 2004, id: 1 }
        })?;

        // Parse content
        if content.is_empty() {
            return Ok(Vec::new());
        }

        let memory_items: Vec<WorkingMemoryItem> = serde_json::from_str(&content).map_err(|e| {
            debug!("Error parsing JSON: {}", e);
            SystemError::MemoryError { code: 2005, id: 1 }
        })?;

        // Get last n messages
        let mut messages: Vec<PromptMessage> = memory_items
            .iter()
            .rev()
            .take(last_n as usize)
            .map(|item| item.message.clone())
            .collect();

        messages.reverse();

        Ok(messages)
    }
}

impl InitialiseMemory for WorkingMemoryDiskRepository {
    async fn initialise(&self) -> SystemResult<()> {
        let id = &self.agent_id;
        let config = &self.config;
        let disk_config = &config.disk.clone().unwrap();
        let path_str = format!("{}/{}", disk_config.path, id);

        let path = Path::new(path_str.as_str());
        if path.exists() {
            debug!("Directory exists!");
        } else {
            fs::create_dir_all(path_str)
                .map_err(|e| {
                    println!("Error path for file {}.", e);
                    SystemError::MemoryError { code: 2006, id: 1 }
                })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use crate::llm::prompt::{PromptMessage, UserContent};
    use crate::memory::memory::{InitialiseMemory, MemoryContext};
    use crate::memory::repository::working_disk_repository::WorkingMemoryDiskRepository;
    use crate::memory::repository::working_repository::{WorkingMemoryCommand, WorkingMemoryConfiguration, WorkingMemoryDiskRepositoryConfiguration, WorkingMemoryQuery};


    struct TestFixture {
        id: String,
        path: String,
        path_str: String,
    }

    impl TestFixture {

        fn new() -> Self {
            let id = "unittest-1".to_string();
            let path = "./tmp".to_string();
            let path_str = format!("{}/{}", path, id);
            Self {
                id, path, path_str,
            }
        }
    }

    impl Drop for TestFixture {

        fn drop(&mut self) {

            let path = Path::new(self.path_str.as_str());
            let _ = fs::remove_dir_all(path);
        }
    }

    #[tokio::test]
    async fn test_working_memory_save_and_read_message() {

        let fixture = TestFixture::new();

        let id = fixture.id.as_str();
        let path = fixture.path.as_str();

        let config = WorkingMemoryConfiguration {
            id: id.to_string(),
            r#type: "disk".to_string(),
            disk: Some(WorkingMemoryDiskRepositoryConfiguration {
                path: path.to_string(),
            }),
            arango: None,
        };
        let repository = WorkingMemoryDiskRepository::new(&config).await.unwrap();

        let _ = repository.initialise().await;

        let prompt = PromptMessage::System {
            name: Some("system".to_string()),
            content: String::from("Test system"),
            role: "system".to_string(),
        };

        let context = MemoryContext {
             interaction_id: "ut".to_string(),
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

