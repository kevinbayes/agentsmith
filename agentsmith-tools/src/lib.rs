//! Tool trait, process registry, and builtin tools.
//!
//! File tools open from a workdir directory fd (`openat2` + `RESOLVE_BENEATH`)
//! and refuse a path that resolves outside that directory. `shell` is not
//! jailed: confinement is the operator's sandbox around the process.

mod builtin;
mod registry;
mod tool;

pub use agentsmith_core::{Error, ToolErrorKind};
pub use builtin::{
    register_builtins, BuiltinLimits, Finish, ListDir, ReadFile, Shell, WriteFile,
    DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_MAX_READ_BYTES, DEFAULT_MAX_WRITE_BYTES,
    DEFAULT_SHELL_TIMEOUT_SECS, MAX_DIR_ENTRIES, MAX_OUTPUT_BYTES_CEILING, MAX_READ_BYTES_CEILING,
    MAX_SHELL_TIMEOUT_SECS, MAX_WRITE_BYTES_CEILING,
};
pub use registry::ToolRegistry;
pub use tokio_util::sync::CancellationToken;
pub use tool::{Tool, ToolCtx, ToolDefinition, ToolOutput};
