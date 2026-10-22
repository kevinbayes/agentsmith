use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use serde::Deserialize;
use serde_json::{json, Value};

use agentsmith_core::{Error, ToolErrorKind};

use super::{
    clamp_output_bytes, clamp_write_bytes, join_blocking, open_error, resolve_under_workdir,
    DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_MAX_WRITE_BYTES,
};
use crate::tool::{ensure_not_cancelled, tool_error, Tool, ToolCtx, ToolDefinition, ToolOutput};

const NAME: &str = "write_file";

/// Create or replace a UTF-8 file that stays inside the run workdir.
#[derive(Debug)]
pub struct WriteFile {
    max_write_bytes: u64,
    max_output_bytes: u64,
}

impl Default for WriteFile {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_WRITE_BYTES, DEFAULT_MAX_OUTPUT_BYTES)
    }
}

impl WriteFile {
    /// `max_write_bytes` is clamped to 1 MiB. Oversized contents are refused
    /// before the file is created or truncated.
    pub fn new(max_write_bytes: u64, max_output_bytes: u64) -> Self {
        Self {
            max_write_bytes: clamp_write_bytes(max_write_bytes),
            max_output_bytes: clamp_output_bytes(max_output_bytes),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    path: String,
    contents: String,
}

#[async_trait::async_trait]
impl Tool for WriteFile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: NAME.to_string(),
            description: "Create or replace a UTF-8 file under the working directory.".to_string(),
            parameters: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["path", "contents"],
                "properties": {
                    "path": { "type": "string" },
                    "contents": { "type": "string" }
                }
            }),
        }
    }

    async fn call(&self, ctx: ToolCtx, arguments: Value) -> Result<ToolOutput, Error> {
        ensure_not_cancelled(NAME, &ctx)?;
        let args: Args = super::parse_args(NAME, arguments)?;
        if args.contents.len() as u64 > self.max_write_bytes {
            return Err(tool_error(
                NAME,
                ToolErrorKind::Limit,
                format!("contents exceed max_write_bytes ({})", self.max_write_bytes),
            ));
        }
        let path = resolve_under_workdir(NAME, &ctx.workdir, &args.path, true)?;
        ensure_not_cancelled(NAME, &ctx)?;
        let max_output = self.max_output_bytes;
        let nbytes = args.contents.len();
        join_blocking(
            NAME,
            tokio::task::spawn_blocking(move || write_nofollow(&path, &args.contents)),
        )
        .await?;
        let message = format!("wrote {nbytes} bytes");
        Ok(ToolOutput {
            content: super::cap_output(&message, max_output, false),
        })
    }
}

fn write_nofollow(path: &Path, contents: &str) -> Result<(), Error> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => return Err(super::escape(NAME)),
        Ok(meta) if !meta.is_file() => {
            return Err(tool_error(
                NAME,
                ToolErrorKind::Execution,
                "not a regular file",
            ));
        }
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            if let Some(parent) = path.parent() {
                if !parent.is_dir() {
                    return Err(tool_error(
                        NAME,
                        ToolErrorKind::Execution,
                        "parent directory does not exist",
                    ));
                }
            }
        }
        Err(err) => {
            return Err(tool_error(
                NAME,
                ToolErrorKind::Execution,
                format!("path: {err}"),
            ));
        }
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|err| open_error(NAME, err))?;
    file.write_all(contents.as_bytes())
        .map_err(|err| tool_error(NAME, ToolErrorKind::Execution, format!("write: {err}")))?;
    Ok(())
}
