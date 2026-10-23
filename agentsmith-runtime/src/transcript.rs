use std::path::Path;
use std::sync::Arc;

use agentsmith_core::Error;
use agentsmith_llm::Message;

use crate::jsonl::JsonlWriter;
use crate::loop_::{Phase, RunEvents, RunId, RunStatus};

/// In-memory conversation plus an optional append-only JSONL log.
#[derive(Default)]
pub struct Transcript {
    messages: Vec<Message>,
    log: Option<JsonlWriter>,
    /// Armed by a loop that owns a log. Unset for `Transcript::empty`.
    finish_hook: Option<FinishHook>,
    /// `finish` already attempted a terminal line. Drop must not append another.
    closed: bool,
}

struct FinishHook {
    events: Arc<dyn RunEvents>,
    run_id: RunId,
    phase: Phase,
}

impl Transcript {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn with_log(data_dir: &Path, run_id: RunId) -> Result<Self, Error> {
        Ok(Self {
            messages: Vec::new(),
            log: Some(JsonlWriter::create(data_dir, run_id)?),
            finish_hook: None,
            closed: false,
        })
    }

    /// Panic unwind writes one terminal line when `finish` has not.
    pub fn arm_unwind(&mut self, events: Arc<dyn RunEvents>, run_id: RunId, phase: Phase) {
        self.finish_hook = Some(FinishHook {
            events,
            run_id,
            phase,
        });
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
        phase: Phase,
    ) -> Result<(), Error> {
        if self.closed {
            return Ok(());
        }
        // A failed append must not leave Drop free to add a second line.
        self.closed = true;
        if let Some(log) = &mut self.log {
            log.finish(status, error_kind, iteration, phase)?;
        }
        Ok(())
    }

    pub fn into_messages(mut self) -> Vec<Message> {
        self.closed = true;
        std::mem::take(&mut self.messages)
    }
}

impl Drop for Transcript {
    fn drop(&mut self) {
        if self.closed || !std::thread::panicking() {
            return;
        }
        let Some(hook) = &self.finish_hook else {
            return;
        };
        // Publish first. A join that already recorded `timed_out` must win.
        let (status, error_kind) =
            hook.events
                .record_terminal(hook.run_id, RunStatus::Failed, Some("panic".to_string()));
        self.closed = true;
        if let Some(log) = &mut self.log {
            let _ = log.finish(status, error_kind.as_deref(), 0, hook.phase);
        }
    }
}
