use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use agentsmith_llm::Message;
use serde_json::{json, Value};

use agentsmith_core::Error;

use crate::loop_::{RunId, RunStatus};

/// Terminal JSONL files kept after a run finishes. Active files are not in this cap.
const TERMINAL_JSONL_KEEP: usize = 20;

pub struct JsonlWriter {
    path: PathBuf,
    file: File,
    run_id: RunId,
}

impl JsonlWriter {
    pub fn create(data_dir: &Path, run_id: RunId) -> Result<Self, Error> {
        let dir = data_dir.join("runs");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{run_id}.jsonl"));
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self { path, file, run_id })
    }

    pub fn append_message(&mut self, message: &Message) -> Result<(), Error> {
        self.write_value(&message_value(message))
    }

    pub fn finish(
        &mut self,
        status: RunStatus,
        error_kind: Option<&str>,
        iteration: u32,
    ) -> Result<(), Error> {
        self.write_value(&json!({
            "event": "finished",
            "run_id": self.run_id.to_string(),
            "status": status,
            "phase": "react",
            "iteration": iteration,
            "error_kind": error_kind,
        }))?;
        if let Some(dir) = self.path.parent() {
            // One finisher. A rotation failure must not discard the finished run.
            let _ = rotate_terminal_jsonl(dir, TERMINAL_JSONL_KEEP);
        }
        Ok(())
    }

    fn write_value(&mut self, value: &Value) -> Result<(), Error> {
        let line =
            serde_json::to_string(value).map_err(|err| Error::Loop(format!("jsonl: {err}")))?;
        self.file.write_all(line.as_bytes())?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        Ok(())
    }
}

fn message_value(message: &Message) -> Value {
    match message {
        Message::System { content } => json!({"role": "system", "content": content}),
        Message::User { content } => json!({"role": "user", "content": content}),
        Message::Assistant {
            content,
            tool_calls,
        } => json!({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls.iter().map(|call| json!({
                "id": call.id,
                "name": call.name,
                "arguments_raw": call.arguments_raw,
                "arguments": call.arguments,
            })).collect::<Vec<_>>()
        }),
        Message::Tool {
            tool_call_id,
            content,
        } => json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }),
    }
}

/// Delete the oldest terminal `*.jsonl` files until `keep` remain.
/// A file whose last line is not a `finished` event is left alone.
pub fn rotate_terminal_jsonl(dir: &Path, keep: usize) -> Result<(), Error> {
    if !dir.is_dir() {
        return Ok(());
    }
    let mut terminal = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
            continue;
        }
        if !is_terminal(&path) {
            continue;
        }
        let modified = match entry.metadata().and_then(|meta| meta.modified()) {
            Ok(modified) => modified,
            Err(_) => continue,
        };
        terminal.push((modified, path));
    }
    if terminal.len() <= keep {
        return Ok(());
    }
    terminal.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let excess = terminal.len() - keep;
    for (_, path) in terminal.into_iter().take(excess) {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

fn is_terminal(path: &Path) -> bool {
    let Some(line) = last_line(path) else {
        return false;
    };
    serde_json::from_str::<Value>(&line)
        .ok()
        .and_then(|value| {
            value
                .get("event")
                .and_then(Value::as_str)
                .map(|event| event == "finished")
        })
        .unwrap_or(false)
}

/// Last non-empty line. Reads the tail so a long transcript is not loaded.
fn last_line(path: &Path) -> Option<String> {
    let mut file = File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    if len == 0 {
        return None;
    }
    let take = len.min(8192);
    let take_usize = usize::try_from(take).ok()?;
    file.seek(SeekFrom::End(-i64::try_from(take).ok()?)).ok()?;
    let mut buf = vec![0u8; take_usize];
    file.read_exact(&mut buf).ok()?;
    if len > take {
        if let Some(pos) = buf.iter().position(|byte| *byte == b'\n') {
            buf.drain(..=pos);
        }
    }
    while buf.last() == Some(&b'\n') {
        buf.pop();
    }
    let start = buf
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map(|index| index + 1)
        .unwrap_or(0);
    let line = String::from_utf8(buf[start..].to_vec()).ok()?;
    if line.is_empty() {
        None
    } else {
        Some(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "agentsmith-jsonl-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_at(path: &Path, body: &str, secs: u64) {
        std::fs::write(path, body).unwrap();
        let file = OpenOptions::new().write(true).open(path).unwrap();
        file.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs))
            .unwrap();
    }

    #[test]
    fn rotation_keeps_twenty_terminal_files_and_the_active_one() {
        let dir = scratch();
        write_at(
            &dir.join("active.jsonl"),
            "{\"role\":\"user\",\"content\":\"still running\"}\n",
            1,
        );
        write_at(&dir.join("notes.txt"), "keep me\n", 1);
        write_at(&dir.join("weird.jsonl"), "not json\n", 1);
        for index in 0..21 {
            write_at(
                &dir.join(format!("t-{index:02}.jsonl")),
                "{\"event\":\"finished\",\"status\":\"succeeded\",\"error_kind\":null,\"iteration\":0,\"phase\":\"react\"}\n",
                10 + index,
            );
        }
        rotate_terminal_jsonl(&dir, TERMINAL_JSONL_KEEP).unwrap();
        assert!(dir.join("active.jsonl").is_file());
        assert!(dir.join("notes.txt").is_file());
        assert!(dir.join("weird.jsonl").is_file());
        assert!(
            !dir.join("t-00.jsonl").is_file(),
            "oldest terminal file should be deleted"
        );
        assert!(dir.join("t-01.jsonl").is_file());
        assert!(dir.join("t-20.jsonl").is_file());
        let terminal = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| is_terminal(&entry.path()))
            .count();
        assert_eq!(terminal, 20);
        let _ = std::fs::remove_dir_all(dir);
    }
}
