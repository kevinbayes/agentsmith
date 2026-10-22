use std::path::Path;

use agentsmith_core::Error;
use agentsmith_llm::Message;

use crate::jsonl::JsonlWriter;
use crate::loop_::{RunId, RunStatus};

/// In-memory conversation plus an optional append-only JSONL log.
#[derive(Default)]
pub struct Transcript {
    messages: Vec<Message>,
    log: Option<JsonlWriter>,
}

impl Transcript {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn with_log(data_dir: &Path, run_id: RunId) -> Result<Self, Error> {
        Ok(Self {
            messages: Vec::new(),
            log: Some(JsonlWriter::create(data_dir, run_id)?),
        })
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn push(&mut self, message: Message) -> Result<(), Error> {
        if let Some(log) = &mut self.log {
            log.append_message(&message)?;
        }
        self.messages.push(message);
        Ok(())
    }

    pub fn finish(
        &mut self,
        status: RunStatus,
        error_kind: Option<&str>,
        iteration: u32,
    ) -> Result<(), Error> {
        if let Some(log) = &mut self.log {
            log.finish(status, error_kind, iteration)?;
        }
        Ok(())
    }

    pub fn into_messages(self) -> Vec<Message> {
        self.messages
    }
}
