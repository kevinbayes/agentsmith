//! Builtins share one path jail and one output cap.
//!
//! `read_file`, `write_file`, and `list_dir` resolve `..` and symlinks and
//! require the result to stay under the workdir. A symlink that points outside
//! is an execution error before any bytes are read or written. `shell` does
//! not use this jail.

mod finish;
mod list_dir;
mod read_file;
mod shell;
mod write_file;

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde_json::Value;

use agentsmith_core::{Error, ToolErrorKind};

use crate::registry::ToolRegistry;
use crate::tool::tool_error;

pub use finish::Finish;
pub use list_dir::ListDir;
pub use read_file::ReadFile;
pub use shell::Shell;
pub use write_file::WriteFile;

pub const DEFAULT_MAX_READ_BYTES: u64 = 64 * 1024;
pub const MAX_READ_BYTES_CEILING: u64 = 256 * 1024;
pub const DEFAULT_MAX_WRITE_BYTES: u64 = 256 * 1024;
pub const MAX_WRITE_BYTES_CEILING: u64 = 1024 * 1024;
pub const DEFAULT_MAX_OUTPUT_BYTES: u64 = 32 * 1024;
pub const MAX_OUTPUT_BYTES_CEILING: u64 = 32 * 1024;
pub const DEFAULT_SHELL_TIMEOUT_SECS: u64 = 30;
pub const MAX_SHELL_TIMEOUT_SECS: u64 = 120;
pub const MAX_DIR_ENTRIES: usize = 500;

const TRUNCATED_SUFFIX: &str = "\n[truncated]";

/// Limits passed to [`register_builtins`]. Constructors clamp to the ceilings.
#[derive(Clone, Debug)]
pub struct BuiltinLimits {
    pub max_read_bytes: u64,
    pub max_write_bytes: u64,
    pub max_output_bytes: u64,
    pub shell_timeout: Duration,
    pub shell_max_timeout: Duration,
    /// Shell is host execution. Callers turn it on; the default stays off.
    pub enable_shell: bool,
}

impl Default for BuiltinLimits {
    fn default() -> Self {
        Self {
            max_read_bytes: DEFAULT_MAX_READ_BYTES,
            max_write_bytes: DEFAULT_MAX_WRITE_BYTES,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            shell_timeout: Duration::from_secs(DEFAULT_SHELL_TIMEOUT_SECS),
            shell_max_timeout: Duration::from_secs(MAX_SHELL_TIMEOUT_SECS),
            enable_shell: false,
        }
    }
}

/// Register the always-on builtins, plus `shell` when `limits.enable_shell`.
pub fn register_builtins(registry: &ToolRegistry, limits: &BuiltinLimits) -> Result<(), Error> {
    registry.register(Arc::new(ReadFile::new(
        limits.max_read_bytes,
        limits.max_output_bytes,
    )))?;
    registry.register(Arc::new(WriteFile::new(
        limits.max_write_bytes,
        limits.max_output_bytes,
    )))?;
    registry.register(Arc::new(ListDir::new(limits.max_output_bytes)))?;
    registry.register(Arc::new(Finish::new(limits.max_output_bytes)))?;
    if limits.enable_shell {
        registry.register(Arc::new(Shell::new(
            limits.shell_timeout,
            limits.shell_max_timeout,
            limits.max_output_bytes,
        )))?;
    }
    Ok(())
}

pub(crate) fn clamp_read_bytes(value: u64) -> u64 {
    value.min(MAX_READ_BYTES_CEILING)
}

pub(crate) fn clamp_write_bytes(value: u64) -> u64 {
    value.min(MAX_WRITE_BYTES_CEILING)
}

pub(crate) fn clamp_output_bytes(value: u64) -> u64 {
    value.min(MAX_OUTPUT_BYTES_CEILING)
}

pub(crate) fn parse_args<T: DeserializeOwned>(tool: &str, arguments: Value) -> Result<T, Error> {
    let arguments = match arguments {
        Value::String(raw) => serde_json::from_str(&raw).map_err(|err| {
            tool_error(
                tool,
                ToolErrorKind::InvalidArgs,
                format!("invalid JSON arguments: {err}"),
            )
        })?,
        other => other,
    };
    // A one-field struct deserializes from a one-element array. Every builtin
    // schema is an object, so reject anything else before that happens.
    if !arguments.is_object() {
        return Err(tool_error(
            tool,
            ToolErrorKind::InvalidArgs,
            "arguments must be a JSON object",
        ));
    }
    serde_json::from_value(arguments).map_err(|err| {
        tool_error(
            tool,
            ToolErrorKind::InvalidArgs,
            format!("invalid arguments: {err}"),
        )
    })
}

/// Cut `content` to the output cap. Truncation keeps a `\n[truncated]` suffix
/// when it fits, on a char boundary. `already_cut` forces the suffix even when
/// the remaining text is under the cap (the read cap, or a long directory).
pub(crate) fn cap_output(content: &str, max_bytes: u64, already_cut: bool) -> String {
    let max_bytes = usize::try_from(clamp_output_bytes(max_bytes)).unwrap_or(usize::MAX);
    if !already_cut && content.len() <= max_bytes {
        return content.to_string();
    }
    if already_cut && content.len().saturating_add(TRUNCATED_SUFFIX.len()) <= max_bytes {
        let mut out = String::with_capacity(content.len() + TRUNCATED_SUFFIX.len());
        out.push_str(content);
        out.push_str(TRUNCATED_SUFFIX);
        return out;
    }
    if max_bytes >= TRUNCATED_SUFFIX.len() {
        let end = floor_char_boundary(content, max_bytes - TRUNCATED_SUFFIX.len());
        let mut out = String::with_capacity(end + TRUNCATED_SUFFIX.len());
        out.push_str(&content[..end]);
        out.push_str(TRUNCATED_SUFFIX);
        return out;
    }
    let end = floor_char_boundary(content, max_bytes);
    content[..end].to_string()
}

fn floor_char_boundary(content: &str, max_bytes: usize) -> usize {
    let mut end = max_bytes.min(content.len());
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    end
}

pub(crate) async fn join_blocking<T>(
    tool: &str,
    handle: tokio::task::JoinHandle<Result<T, Error>>,
) -> Result<T, Error> {
    match handle.await {
        Ok(result) => result,
        Err(err) if err.is_panic() => {
            Err(tool_error(tool, ToolErrorKind::Execution, "task panicked"))
        }
        Err(err) => Err(tool_error(tool, ToolErrorKind::Execution, err.to_string())),
    }
}

/// Resolve `requested` against `workdir`. The returned path is canonical when
/// it already exists. A missing final component is returned only when
/// `allow_absent_leaf` is set, and only if every parent stays inside the jail.
pub(crate) fn resolve_under_workdir(
    tool: &str,
    workdir: &Path,
    requested: &str,
    allow_absent_leaf: bool,
) -> Result<PathBuf, Error> {
    if requested.is_empty() || requested.contains('\0') {
        return Err(tool_error(
            tool,
            ToolErrorKind::InvalidArgs,
            "path is empty",
        ));
    }
    let root = std::fs::canonicalize(workdir)
        .map_err(|err| tool_error(tool, ToolErrorKind::Execution, format!("workdir: {err}")))?;
    let requested_path = Path::new(requested);
    let joined = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        root.join(requested_path)
    };
    let lexical = lexical_normalize(&joined);
    if !lexical.starts_with(&root) {
        return Err(escape(tool));
    }
    let Ok(relative) = lexical.strip_prefix(&root) else {
        return Err(escape(tool));
    };

    let mut current = root.clone();
    let parts: Vec<Component<'_>> = relative.components().collect();
    for (index, component) in parts.iter().enumerate() {
        let Component::Normal(name) = component else {
            return Err(escape(tool));
        };
        let next = current.join(name);
        let last = index + 1 == parts.len();
        match std::fs::symlink_metadata(&next) {
            Ok(meta) if meta.file_type().is_symlink() => {
                // Canonicalize before touching anything past the link. A target
                // outside the workdir is an error, not a partial read.
                let target = std::fs::canonicalize(&next).map_err(|_| escape(tool))?;
                if !target.starts_with(&root) {
                    return Err(escape(tool));
                }
                current = target;
            }
            Ok(_) => current = next,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound && last && allow_absent_leaf => {
                current = next;
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(tool_error(
                    tool,
                    ToolErrorKind::Execution,
                    format!("path not found: {requested}"),
                ));
            }
            Err(err) => {
                return Err(tool_error(
                    tool,
                    ToolErrorKind::Execution,
                    format!("path: {err}"),
                ));
            }
        }
    }

    if current.symlink_metadata().is_ok() {
        let canonical = std::fs::canonicalize(&current).map_err(|_| escape(tool))?;
        if !canonical.starts_with(&root) {
            return Err(escape(tool));
        }
        return Ok(canonical);
    }
    if !current.starts_with(&root) {
        return Err(escape(tool));
    }
    Ok(current)
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) => {}
            Component::RootDir => out = PathBuf::from("/"),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(name) => out.push(name),
        }
    }
    out
}

pub(crate) fn escape(tool: &str) -> Error {
    tool_error(
        tool,
        ToolErrorKind::Execution,
        "path escapes the working directory",
    )
}

pub(crate) fn open_error(tool: &str, err: std::io::Error) -> Error {
    // O_NOFOLLOW surfaces a swapped-in symlink as ELOOP. Treat that as a jail
    // failure so the caller does not see a partial read or a followed write.
    if err.raw_os_error() == Some(libc::ELOOP) {
        escape(tool)
    } else {
        tool_error(tool, ToolErrorKind::Execution, format!("path: {err}"))
    }
}
