use std::fs::File;
use std::io::Read;
use std::os::fd::AsRawFd;

use serde::Deserialize;
use serde_json::{json, Value};

use agentsmith_core::Error;

use super::{
    cap_output, clamp_output_bytes, fstat_fd, join_blocking, open_error, WorkDirFd,
    DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_MAX_READ_BYTES, MAX_READ_BYTES_CEILING,
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
            max_read_bytes: max_read_bytes.min(MAX_READ_BYTES_CEILING),
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
        ensure_not_cancelled(NAME, &ctx)?;
        let workdir = ctx.workdir.clone();
        let requested = args.path;
        let max_read = self.max_read_bytes;
        let max_output = self.max_output_bytes;
        // Open and read inside one blocking task so nothing awaits between
        // the directory-fd walk and the read.
        let (text, truncated) = join_blocking(
            NAME,
            tokio::task::spawn_blocking(move || read_limited(&workdir, &requested, max_read)),
        )
        .await?;
        Ok(ToolOutput {
            content: cap_output(&text, max_output, truncated),
        })
    }
}

fn read_limited(
    workdir: &std::path::Path,
    requested: &str,
    max_read: u64,
) -> Result<(String, bool), Error> {
    let root = WorkDirFd::open(NAME, workdir)?;
    let relative = root.relative(NAME, requested)?;
    let fd = root
        .open_relative(
            &relative,
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NONBLOCK,
        )
        .map_err(|err| {
            if err.raw_os_error() == Some(libc::ENXIO) {
                return tool_error(NAME, ToolErrorKind::Execution, "not a regular file");
            }
            open_error(NAME, requested, err)
        })?;
    let st = fstat_fd(fd.as_raw_fd())
        .map_err(|err| tool_error(NAME, ToolErrorKind::Execution, format!("path: {err}")))?;
    if st.st_mode & libc::S_IFMT != libc::S_IFREG {
        return Err(tool_error(
            NAME,
            ToolErrorKind::Execution,
            "not a regular file",
        ));
    }
    // A FIFO would have failed the type check. Regular-file reads do not need
    // the nonblocking flag that kept `open` from waiting on a pipe.
    let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
    if flags < 0 {
        let err = std::io::Error::last_os_error();
        return Err(tool_error(
            NAME,
            ToolErrorKind::Execution,
            format!("path: {err}"),
        ));
    }
    if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, flags & !libc::O_NONBLOCK) } < 0 {
        let err = std::io::Error::last_os_error();
        return Err(tool_error(
            NAME,
            ToolErrorKind::Execution,
            format!("path: {err}"),
        ));
    }
    let mut file = File::from(fd);
    let mut buf = Vec::new();
    (&mut file)
        .take(max_read)
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
