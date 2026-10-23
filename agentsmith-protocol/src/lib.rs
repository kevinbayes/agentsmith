//! AHSP 1.0 documents this process sends.
//!
//! The checked-in schema is the closed producer schema. Servers that accept a
//! later minor of major 1 must ignore unknown fields; this crate does not.

use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use uuid::Uuid;

pub const PROTOCOL: &str = "ahsp/1.0";

/// Authoritative protocol id. Serializes as `ahsp/1.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtocolId;

impl Serialize for ProtocolId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(PROTOCOL)
    }
}

impl<'de> Deserialize<'de> for ProtocolId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        if text == PROTOCOL {
            Ok(ProtocolId)
        } else {
            Err(serde::de::Error::custom(format!(
                "unsupported protocol {text}"
            )))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateReport {
    pub protocol: ProtocolId,
    pub report_id: Uuid,
    pub sequence: u64,
    #[serde(with = "millis")]
    pub sent_at: DateTime<Utc>,
    pub harness: HarnessIdentity,
    pub host: HostState,
    pub runs: Vec<RunReport>,
    pub tools: Vec<ToolReport>,
    pub mcp_servers: Vec<McpServerReport>,
    pub limits: LimitsReport,
    /// Omitted when false. One bit covers every trim.
    #[serde(default, skip_serializing_if = "is_false")]
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessIdentity {
    pub instance_id: Uuid,
    pub name: String,
    pub version: String,
    pub pid: u32,
    pub process_mode: ProcessMode,
    pub uptime_secs: u64,
    pub stopping: bool,
    pub config_fingerprint: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessMode {
    Foreground,
    Service,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostState {
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub processes: Vec<ProcessWatch>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessWatch {
    pub name: String,
    pub binary: String,
    pub installed: bool,
    pub version: Option<String>,
    pub path: Option<String>,
    pub running_instances: Vec<RunningInstance>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunningInstance {
    pub pid: u32,
    pub name: String,
    pub cmd_summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunReport {
    pub run_id: Uuid,
    pub parent_run_id: Option<Uuid>,
    pub root_run_id: Uuid,
    pub depth: u32,
    pub name: String,
    pub mode: RunMode,
    pub status: RunStatus,
    pub phase: Phase,
    pub iteration: u32,
    pub max_iterations: u32,
    pub task_sha256_12: String,
    pub task_title: String,
    pub spec_id: Option<String>,
    #[serde(with = "millis")]
    pub started_at: DateTime<Utc>,
    #[serde(with = "millis_opt")]
    pub finished_at: Option<DateTime<Utc>>,
    pub error_kind: Option<String>,
    pub children: Vec<ChildRef>,
    pub tool_use: std::collections::BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChildRef {
    pub run_id: Uuid,
    pub name: String,
    pub status: RunStatus,
    pub depth: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    React,
    Plan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Succeeded,
    Failed,
    Partial,
    Cancelled,
    MaxIterations,
    TimedOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Plan,
    Execute,
    React,
    Join,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolReport {
    pub name: String,
    pub source: ToolSource,
    /// Present only when `source` is `mcp`. Omitted for builtins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server: Option<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSource {
    Builtin,
    Mcp,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerReport {
    pub name: String,
    pub transport: McpTransport,
    pub command: String,
    pub status: McpStatus,
    pub required: bool,
    pub tool_count: u32,
    pub tools_skipped: u32,
    pub restart_count: u32,
    #[serde(with = "millis_opt")]
    pub connected_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum McpTransport {
    #[serde(rename = "stdio")]
    Stdio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpStatus {
    Up,
    Down,
    Connecting,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LimitsReport {
    pub max_subagent_depth: u32,
    pub max_fanout: u32,
    pub live_subagents: u32,
}

/// Ack fields this process reads. Unknown fields, including `commands`, are not
/// declared here so serde ignores them. `commands` is stripped before this
/// deserialize; nothing in the ack is executed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Ack {
    pub protocol: Option<String>,
    pub accepted: Option<bool>,
    pub report_id: Option<Uuid>,
    pub server_time: Option<DateTime<Utc>>,
    pub next_interval_secs: Option<u64>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

mod millis {
    use super::*;

    pub fn serialize<S>(dt: &DateTime<Utc>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&dt.to_rfc3339_opts(SecondsFormat::Millis, true))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<DateTime<Utc>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        parse_millis(&text).map_err(serde::de::Error::custom)
    }
}

mod millis_opt {
    use super::*;

    pub fn serialize<S>(dt: &Option<DateTime<Utc>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match dt {
            Some(dt) => millis::serialize(dt, serializer),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<DateTime<Utc>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Option::<String>::deserialize(deserializer)?;
        match value {
            None => Ok(None),
            Some(text) => parse_millis(&text)
                .map(Some)
                .map_err(serde::de::Error::custom),
        }
    }
}

fn parse_millis(text: &str) -> Result<DateTime<Utc>, chrono::ParseError> {
    DateTime::parse_from_rfc3339(text).map(|dt| dt.with_timezone(&Utc))
}
