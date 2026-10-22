use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use serde::Deserialize;
use serde_json::{json, Value};

use agentsmith_core::{Error, ToolErrorKind};

use super::{
    cap_output, clamp_output_bytes, join_blocking, open_error, resolve_under_workdir,
    DEFAULT_MAX_OUTPUT_BYTES, MAX_DIR_ENTRIES,
};
use crate::tool::{ensure_not_cancelled, tool_error, Tool, ToolCtx, ToolDefinition, ToolOutput};

const NAME: &str = "list_dir";

/// Non-recursive listing of one directory inside the run workdir.
#[derive(Debug)]
pub struct ListDir {
    max_output_bytes: u64,
}

impl Default for ListDir {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_OUTPUT_BYTES)
    }
}

impl ListDir {
    pub fn new(max_output_bytes: u64) -> Self {
        Self {
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
impl Tool for ListDir {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: NAME.to_string(),
            description: "List one directory under the working directory. Not recursive."
                .to_string(),
            parameters: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["path"],
                "properties": {
                    "path": { "type": "string" }
                }
            }),
        }
    }

    async fn call(&self, ctx: ToolCtx, arguments: Value) -> Result<ToolOutput, Error> {
        ensure_not_cancelled(NAME, &ctx)?;
        let args: Args = super::parse_args(NAME, arguments)?;
        let path = resolve_under_workdir(NAME, &ctx.workdir, &args.path, false)?;
        ensure_not_cancelled(NAME, &ctx)?;
        let max_output = self.max_output_bytes;
        let (body, truncated) = join_blocking(
            NAME,
            tokio::task::spawn_blocking(move || list_limited(&path)),
        )
        .await?;
        Ok(ToolOutput {
            content: cap_output(&body, max_output, truncated),
        })
    }
}

fn list_limited(path: &Path) -> Result<(String, bool), Error> {
    // Hold the directory fd open and list through it. O_NOFOLLOW means a
    // symlink swapped in after the jail check is not followed.
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|err| open_error(NAME, err))?;
    let meta = file
        .metadata()
        .map_err(|err| tool_error(NAME, ToolErrorKind::Execution, format!("path: {err}")))?;
    if !meta.is_dir() {
        return Err(tool_error(
            NAME,
            ToolErrorKind::Execution,
            "not a directory",
        ));
    }
    let proc_path = format!("/proc/self/fd/{}", file.as_raw_fd());
    let mut names = Vec::new();
    for entry in std::fs::read_dir(&proc_path)
        .map_err(|err| tool_error(NAME, ToolErrorKind::Execution, format!("list: {err}")))?
    {
        let entry = entry
            .map_err(|err| tool_error(NAME, ToolErrorKind::Execution, format!("list: {err}")))?;
        let file_type = entry
            .file_type()
            .map_err(|err| tool_error(NAME, ToolErrorKind::Execution, format!("list: {err}")))?;
        let mut name = entry.file_name().to_string_lossy().into_owned();
        // `file_type` does not follow symlinks, so a link to a directory
        // outside the jail is just a name.
        if file_type.is_dir() {
            name.push('/');
        }
        names.push(name);
    }
    names.sort();
    let truncated = names.len() > MAX_DIR_ENTRIES;
    if truncated {
        names.truncate(MAX_DIR_ENTRIES);
    }
    Ok((names.join("\n"), truncated))
}
