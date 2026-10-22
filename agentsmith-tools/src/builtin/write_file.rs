use std::ffi::CString;
use std::fs::File;
use std::io::Write;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use serde::Deserialize;
use serde_json::{json, Value};

use agentsmith_core::{Error, ToolErrorKind};

use super::{
    cap_output, clamp_output_bytes, escape, fstat_fd, join_blocking, open_error, WorkDirFd,
    DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_MAX_WRITE_BYTES, MAX_WRITE_BYTES_CEILING,
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
            max_write_bytes: max_write_bytes.min(MAX_WRITE_BYTES_CEILING),
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
        ensure_not_cancelled(NAME, &ctx)?;
        let workdir = ctx.workdir.clone();
        let requested = args.path;
        let max_output = self.max_output_bytes;
        let nbytes = args.contents.len();
        join_blocking(
            NAME,
            tokio::task::spawn_blocking(move || {
                write_nofollow(&workdir, &requested, &args.contents)
            }),
        )
        .await?;
        let message = format!("wrote {nbytes} bytes");
        Ok(ToolOutput {
            content: cap_output(&message, max_output, false),
        })
    }
}

fn write_nofollow(workdir: &Path, requested: &str, contents: &str) -> Result<(), Error> {
    let root = WorkDirFd::open(NAME, workdir)?;
    let relative = root.relative(NAME, requested)?;
    let Some(leaf) = relative.file_name().filter(|name| !name.is_empty()) else {
        // `.` or a path that normalizes to the directory itself.
        return match root.open_relative(
            &relative,
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NONBLOCK,
        ) {
            Ok(_) => Err(tool_error(
                NAME,
                ToolErrorKind::Execution,
                "not a regular file",
            )),
            Err(err) => Err(open_error(NAME, requested, err)),
        };
    };
    let leaf = leaf.to_os_string();
    let parent = relative.parent().unwrap_or(Path::new(""));
    let parent_fd = if parent.as_os_str().is_empty() || parent == Path::new(".") {
        let fd = unsafe { libc::fcntl(root.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 0) };
        if fd < 0 {
            let err = std::io::Error::last_os_error();
            return Err(tool_error(
                NAME,
                ToolErrorKind::Execution,
                format!("path: {err}"),
            ));
        }
        // SAFETY: `F_DUPFD_CLOEXEC` returned a new fd, or we already returned.
        unsafe { OwnedFd::from_raw_fd(fd) }
    } else {
        root.open_relative(
            parent,
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NONBLOCK,
        )
        .map_err(|err| {
            if matches!(err.raw_os_error(), Some(libc::ENOENT) | Some(libc::ENOTDIR)) {
                tool_error(
                    NAME,
                    ToolErrorKind::Execution,
                    "parent directory does not exist",
                )
            } else {
                open_error(NAME, requested, err)
            }
        })?
    };

    // The parent fd is already confined. Create or replace only the leaf, and
    // do not follow a symlink that appears as that leaf.
    let c_leaf = CString::new(leaf.as_bytes())
        .map_err(|_| tool_error(NAME, ToolErrorKind::InvalidArgs, "path is empty"))?;
    let flags =
        libc::O_WRONLY | libc::O_CREAT | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK;
    let fd = unsafe { libc::openat(parent_fd.as_raw_fd(), c_leaf.as_ptr(), flags, 0o666) };
    if fd < 0 {
        let err = std::io::Error::last_os_error();
        return Err(match err.raw_os_error() {
            Some(libc::ELOOP) | Some(libc::EXDEV) => escape(NAME),
            Some(libc::EISDIR) | Some(libc::ENXIO) => {
                tool_error(NAME, ToolErrorKind::Execution, "not a regular file")
            }
            _ => open_error(NAME, requested, err),
        });
    }
    // SAFETY: `openat` returned a new fd, or we already returned.
    let owned = unsafe { OwnedFd::from_raw_fd(fd) };
    let st = fstat_fd(owned.as_raw_fd())
        .map_err(|err| tool_error(NAME, ToolErrorKind::Execution, format!("path: {err}")))?;
    if st.st_mode & libc::S_IFMT != libc::S_IFREG {
        return Err(tool_error(
            NAME,
            ToolErrorKind::Execution,
            "not a regular file",
        ));
    }
    let mut file = File::from(owned);
    file.set_len(0)
        .map_err(|err| tool_error(NAME, ToolErrorKind::Execution, format!("write: {err}")))?;
    file.write_all(contents.as_bytes())
        .map_err(|err| tool_error(NAME, ToolErrorKind::Execution, format!("write: {err}")))?;
    Ok(())
}
