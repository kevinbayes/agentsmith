//! Stdio MCP client. One supervisor owns every session and is the only
//! reconnect loop.

mod client;
mod install_tools;
mod stdio;
mod supervisor;

pub use client::{McpClient, McpSessionInfo};
pub use stdio::StdioMcpClient;
pub use supervisor::{McpSupervisor, McpSupervisorConfig};
