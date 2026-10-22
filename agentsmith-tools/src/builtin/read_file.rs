use std::fs::OpenOptions;
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use serde::Deserialize;
use serde_json::{json, Value};

use agentsmith_core::Error;

use super::{
    cap_output, clamp_output_bytes, clamp_read_bytes, join_blocking, open_error,
    resolve_under_workdir, DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_MAX_READ_BYTES,
};
use crate::tool::{ensure_not_cancelled, tool_error, Tool, ToolCtx, ToolDefinition, ToolOutput};
use agentsmith_core::ToolErrorKind;

const NAME: &str = "read_file";

/// Read a UTF-8 file that stays inside the run workdir.
#[derive(Debug)]
pub struct ReadFile {
    max_read_bytes: u64,
    max_output_bytes: u64,
}

impl Default for ReadFile {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_READ_BYTES, DEFAULT_MAX_OUTPUT_BYTES)
    }
}

impl ReadFile {
    /// `max_read_bytes` is clamped to 256 KiB. `max_output_bytes` is clamped to 32 KiB.
    /// Tests pass a small cap; values above the ceiling are not honored.
    pub fn new(max_read_bytes: u64, max_output_bytes: u64) -> Self {
        Self {
            max_read_bytes: clamp_read_bytes(max_read_bytes),
            max_output_bytes: clamp_output_bytes(max_output_bytes),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    path: String,
}

#[async_trait::async_trait]
impl Tool for ReadFile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: NAME.to_string(),
            description:
                "Read a UTF-8 file under the working directory. Refuses paths that escape it."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the working directory."
                    }
                }
            }),
        }
    }

    async fn call(&self, ctx: ToolCtx, arguments: Value) -> Result<ToolOutput, Error> {
        ensure_not_cancelled(NAME, &ctx)?;
        let args: Args = super::parse_args(NAME, arguments)?;
        let path = resolve_under_workdir(NAME, &ctx.workdir, &args.path, false)?;
        ensure_not_cancelled(NAME, &ctx)?;
        let max_read = self.max_read_bytes;
        let max_output = self.max_output_bytes;
        let (text, truncated) = join_blocking(
            NAME,
            tokio::task::spawn_blocking(move || read_limited(&path, max_read)),
        )
        .await?;
        Ok(ToolOutput {
            content: cap_output(&text, max_output, truncated),
        })
    }
}

fn read_limited(path: &Path, max_read: u64) -> Result<(String, bool), Error> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|err| open_error(NAME, err))?;
    let meta = file
        .metadata()
        .map_err(|err| tool_error(NAME, ToolErrorKind::Execution, format!("path: {err}")))?;
    if !meta.is_file() {
        return Err(tool_error(
            NAME,
            ToolErrorKind::Execution,
            "not a regular file",
        ));
    }
    let cap = clamp_read_bytes(max_read);
    let mut buf = Vec::new();
    (&mut file)
        .take(cap)
        .read_to_end(&mut buf)
        .map_err(|err| tool_error(NAME, ToolErrorKind::Execution, format!("read: {err}")))?;
    let mut probe = [0u8; 1];
    let more = file
        .read(&mut probe)
        .map_err(|err| tool_error(NAME, ToolErrorKind::Execution, format!("read: {err}")))?
        == 1;
    let text = decode(&buf, more)?;
    Ok((text, more))
}

fn decode(buf: &[u8], truncated: bool) -> Result<String, Error> {
    match std::str::from_utf8(buf) {
        Ok(text) => Ok(text.to_string()),
        Err(err) if truncated && err.error_len().is_none() => {
            let valid = err.valid_up_to();
            std::str::from_utf8(&buf[..valid])
                .map(|text| text.to_string())
                .map_err(|_| tool_error(NAME, ToolErrorKind::Execution, "file is not valid UTF-8"))
        }
        Err(_) => Err(tool_error(
            NAME,
            ToolErrorKind::Execution,
            "file is not valid UTF-8",
        )),
    }
}
