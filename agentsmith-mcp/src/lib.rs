//! Stdio MCP client. One supervisor owns every session and is the only
//! reconnect loop.

mod install_tools;
mod stdio;
mod supervisor;

pub use supervisor::{McpSnapshot, McpSnapshotStatus, McpSupervisor, McpSupervisorConfig};
