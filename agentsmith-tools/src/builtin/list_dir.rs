use std::ffi::CStr;
use std::os::fd::{AsRawFd, IntoRawFd};
use std::path::Path;

use serde::Deserialize;
use serde_json::{json, Value};

use agentsmith_core::{Error, ToolErrorKind};

use super::{
    cap_output, clamp_output_bytes, fstat_fd, join_blocking, open_error, WorkDirFd,
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
        ensure_not_cancelled(NAME, &ctx)?;
        let workdir = ctx.workdir.clone();
        let requested = args.path;
        let max_output = self.max_output_bytes;
        let (body, truncated) = join_blocking(
            NAME,
            tokio::task::spawn_blocking(move || list_limited(&workdir, &requested)),
        )
        .await?;
        Ok(ToolOutput {
            content: cap_output(&body, max_output, truncated),
        })
    }
}

fn list_limited(workdir: &Path, requested: &str) -> Result<(String, bool), Error> {
    let root = WorkDirFd::open(NAME, workdir)?;
    let relative = root.relative(NAME, requested)?;
    let fd = root
        .open_relative(
            &relative,
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NONBLOCK,
        )
        .map_err(|err| {
            if matches!(err.raw_os_error(), Some(libc::ENOTDIR) | Some(libc::ENXIO)) {
                return tool_error(NAME, ToolErrorKind::Execution, "not a directory");
            }
            open_error(NAME, requested, err)
        })?;
    let st = fstat_fd(fd.as_raw_fd())
        .map_err(|err| tool_error(NAME, ToolErrorKind::Execution, format!("path: {err}")))?;
    if st.st_mode & libc::S_IFMT != libc::S_IFDIR {
        return Err(tool_error(
            NAME,
            ToolErrorKind::Execution,
            "not a directory",
        ));
    }

    let raw = fd.into_raw_fd();
    let dir = unsafe {
        // SAFETY: `raw` is an open directory fd. On success, `fdopendir` owns it
        // and `closedir` closes it. On failure the fd stays ours and we close it.
        libc::fdopendir(raw)
    };
    if dir.is_null() {
        unsafe { libc::close(raw) };
        let err = std::io::Error::last_os_error();
        return Err(tool_error(
            NAME,
            ToolErrorKind::Execution,
            format!("list: {err}"),
        ));
    }
    let _dir = DirClose(dir);

    let mut names = Vec::new();
    let mut truncated = false;
    loop {
        unsafe { *libc::__errno_location() = 0 };
        let ent = unsafe { libc::readdir(dir) };
        if ent.is_null() {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error().unwrap_or(0) != 0 {
                return Err(tool_error(
                    NAME,
                    ToolErrorKind::Execution,
                    format!("list: {err}"),
                ));
            }
            break;
        }
        let dtype = unsafe { (*ent).d_type };
        let name_ptr = unsafe { (*ent).d_name.as_ptr() };
        let name = unsafe { CStr::from_ptr(name_ptr) }
            .to_string_lossy()
            .into_owned();
        if name == "." || name == ".." {
            continue;
        }
        // Keep at most MAX_DIR_ENTRIES names. The next entry proves the cap
        // was exceeded; leave the rest of the directory unread.
        if names.len() == MAX_DIR_ENTRIES {
            truncated = true;
            break;
        }
        let is_dir = if dtype == libc::DT_DIR {
            true
        } else if dtype == libc::DT_UNKNOWN {
            // `d_type` is not filled on some filesystems. Stat the name through
            // the directory fd and do not follow a symlink (a link to a
            // directory must not gain the `/` suffix).
            let dirfd = unsafe { libc::dirfd(dir) };
            if dirfd < 0 {
                let err = std::io::Error::last_os_error();
                return Err(tool_error(
                    NAME,
                    ToolErrorKind::Execution,
                    format!("list: {err}"),
                ));
            }
            let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
            let rc = unsafe {
                libc::fstatat(dirfd, name_ptr, st.as_mut_ptr(), libc::AT_SYMLINK_NOFOLLOW)
            };
            if rc != 0 {
                let err = std::io::Error::last_os_error();
                return Err(tool_error(
                    NAME,
                    ToolErrorKind::Execution,
                    format!("list: {err}"),
                ));
            }
            unsafe { st.assume_init() }.st_mode & libc::S_IFMT == libc::S_IFDIR
        } else {
            false
        };
        if is_dir {
            names.push(format!("{name}/"));
        } else {
            names.push(name);
        }
    }
    names.sort();
    Ok((names.join("\n"), truncated))
}

/// `closedir` the `DIR` from `fdopendir`, including on error returns.
struct DirClose(*mut libc::DIR);

impl Drop for DirClose {
    fn drop(&mut self) {
        unsafe { libc::closedir(self.0) };
    }
}
