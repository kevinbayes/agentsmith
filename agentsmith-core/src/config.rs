use std::collections::BTreeMap;
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};

use crate::id::sha256_12;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::Error;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_LLM_TIMEOUT_SECS: u64 = 120;
const CEILING_LLM_TIMEOUT_SECS: u64 = 300;
const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 4096;
const DEFAULT_LLM_RETRIES: u64 = 2;
const CEILING_LLM_RETRIES: u64 = 4;

const DEFAULT_REACT_MAX_ITERATIONS: u64 = 25;
const CEILING_REACT_MAX_ITERATIONS: u64 = 100;
const DEFAULT_STEP_MAX_ITERATIONS: u64 = 8;
const CEILING_STEP_MAX_ITERATIONS: u64 = 50;
const DEFAULT_PLANNER_ATTEMPTS: u64 = 2;
const CEILING_PLANNER_ATTEMPTS: u64 = 2;
const DEFAULT_MAX_CONSECUTIVE_TOOL_ERRORS: u64 = 5;
const CEILING_MAX_CONSECUTIVE_TOOL_ERRORS: u64 = 20;
const DEFAULT_MAX_SUBAGENT_DEPTH: u64 = 2;
const CEILING_MAX_SUBAGENT_DEPTH: u64 = 4;
const DEFAULT_MAX_FANOUT: u64 = 8;
const CEILING_MAX_FANOUT: u64 = 8;
const DEFAULT_MAX_LIVE_SUBAGENTS: u64 = 16;
const CEILING_MAX_LIVE_SUBAGENTS: u64 = 32;
const DEFAULT_SPAWN_TIMEOUT_SECS: u64 = 120;
const CEILING_SPAWN_TIMEOUT_SECS: u64 = 600;

const DEFAULT_TOOL_MAX_OUTPUT_BYTES: u64 = 32 * 1024;
const CEILING_TOOL_MAX_OUTPUT_BYTES: u64 = 32 * 1024;
const DEFAULT_SHELL_TIMEOUT_SECS: u64 = 30;
const CEILING_SHELL_TIMEOUT_SECS: u64 = 120;
const DEFAULT_SHELL_MAX_TIMEOUT_SECS: u64 = 120;
const CEILING_SHELL_MAX_TIMEOUT_SECS: u64 = 120;
const DEFAULT_MAX_READ_BYTES: u64 = 64 * 1024;
const CEILING_MAX_READ_BYTES: u64 = 256 * 1024;
const DEFAULT_MAX_WRITE_BYTES: u64 = 256 * 1024;
const CEILING_MAX_WRITE_BYTES: u64 = 1024 * 1024;

const DEFAULT_BIND: &str = "127.0.0.1:7433";
const DEFAULT_MAX_RUNS: u64 = 2;
const CEILING_MAX_RUNS: u64 = 8;
const DEFAULT_SHUTDOWN_GRACE_SECS: u64 = 15;
const CEILING_SHUTDOWN_GRACE_SECS: u64 = 60;
const CEILING_RUN_SLOTS: u64 = 64;

const DEFAULT_REPORTER_INTERVAL_SECS: u64 = 30;
const MIN_REPORTER_INTERVAL_SECS: u64 = 10;
const MAX_REPORTER_INTERVAL_SECS: u64 = 300;
const DEFAULT_REPORTER_TIMEOUT_SECS: u64 = 10;
const CEILING_REPORTER_TIMEOUT_SECS: u64 = 30;
const DEFAULT_TOKEN_SKEW_SECS: u64 = 30;
const DEFAULT_MAX_BODY_BYTES: u64 = 256 * 1024;
const CEILING_MAX_BODY_BYTES: u64 = 256 * 1024;

const DEFAULT_MCP_CONNECT_TIMEOUT_SECS: u64 = 10;
const CEILING_MCP_CONNECT_TIMEOUT_SECS: u64 = 60;
const DEFAULT_MCP_CALL_TIMEOUT_SECS: u64 = 60;
const CEILING_MCP_CALL_TIMEOUT_SECS: u64 = 300;
const CEILING_HOST_WATCH: usize = 32;

const ETC_CONFIG: &str = "/etc/agentsmith/config.toml";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FailurePolicy {
    #[default]
    Continue,
    FailFast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    #[default]
    Stdio,
}

impl<'de> Deserialize<'de> for McpTransport {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        if raw == "stdio" {
            Ok(Self::Stdio)
        } else {
            Err(serde::de::Error::custom(
                "streamable HTTP MCP is not supported in v1",
            ))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub harness: HarnessConfig,
    pub llm: LlmConfig,
    #[serde(rename = "loop")]
    pub loop_: LoopConfig,
    pub tools: ToolsConfig,
    pub service: ServiceConfig,
    pub reporter: ReporterConfig,
    pub host: HostConfig,
    pub mcp: McpConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct HarnessConfig {
    pub name: String,
    pub log_level: LogLevel,
    pub data_dir: PathBuf,
    pub workdir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
    pub timeout_secs: u64,
    pub max_output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    pub retries: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoopConfig {
    pub react_max_iterations: u64,
    pub step_max_iterations: u64,
    pub planner_attempts: u64,
    pub max_consecutive_tool_errors: u64,
    pub max_subagent_depth: u64,
    pub max_fanout: u64,
    pub max_live_subagents: u64,
    pub spawn_timeout_secs: u64,
    pub failure_policy: FailurePolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsConfig {
    pub max_output_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow: Option<Vec<String>>,
    pub shell: ShellConfig,
    pub files: FilesConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShellConfig {
    /// `None` when the key is absent. Foreground and service resolve it later.
    pub enabled: Option<bool>,
    pub timeout_secs: u64,
    pub max_timeout_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FilesConfig {
    pub max_read_bytes: u64,
    pub max_write_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServiceConfig {
    pub bind: String,
    pub token: String,
    pub max_runs: u64,
    pub shutdown_grace_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReporterConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub token_url: String,
    pub client_id: String,
    pub client_secret: String,
    pub harness_name: String,
    pub interval_secs: u64,
    pub timeout_secs: u64,
    pub token_skew_secs: u64,
    pub max_body_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct HostConfig {
    pub watch: Vec<HostWatch>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostWatch {
    pub name: String,
    pub binary: String,
    #[serde(default)]
    pub match_node: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct McpConfig {
    pub servers: BTreeMap<String, McpServerConfig>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct McpServerConfig {
    pub transport: McpTransport,
    pub command: String,
    pub args: Vec<String>,
    pub required: bool,
    pub connect_timeout_secs: u64,
    pub call_timeout_secs: u64,
    pub env: BTreeMap<String, String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            model: String::new(),
            api_key: String::new(),
            timeout_secs: DEFAULT_LLM_TIMEOUT_SECS,
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
            temperature: None,
            retries: DEFAULT_LLM_RETRIES,
        }
    }
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            react_max_iterations: DEFAULT_REACT_MAX_ITERATIONS,
            step_max_iterations: DEFAULT_STEP_MAX_ITERATIONS,
            planner_attempts: DEFAULT_PLANNER_ATTEMPTS,
            max_consecutive_tool_errors: DEFAULT_MAX_CONSECUTIVE_TOOL_ERRORS,
            max_subagent_depth: DEFAULT_MAX_SUBAGENT_DEPTH,
            max_fanout: DEFAULT_MAX_FANOUT,
            max_live_subagents: DEFAULT_MAX_LIVE_SUBAGENTS,
            spawn_timeout_secs: DEFAULT_SPAWN_TIMEOUT_SECS,
            failure_policy: FailurePolicy::Continue,
        }
    }
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            max_output_bytes: DEFAULT_TOOL_MAX_OUTPUT_BYTES,
            allow: None,
            shell: ShellConfig::default(),
            files: FilesConfig::default(),
        }
    }
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            enabled: None,
            timeout_secs: DEFAULT_SHELL_TIMEOUT_SECS,
            max_timeout_secs: DEFAULT_SHELL_MAX_TIMEOUT_SECS,
        }
    }
}

impl Default for FilesConfig {
    fn default() -> Self {
        Self {
            max_read_bytes: DEFAULT_MAX_READ_BYTES,
            max_write_bytes: DEFAULT_MAX_WRITE_BYTES,
        }
    }
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            bind: DEFAULT_BIND.to_string(),
            token: String::new(),
            max_runs: DEFAULT_MAX_RUNS,
            shutdown_grace_secs: DEFAULT_SHUTDOWN_GRACE_SECS,
        }
    }
}

impl Default for ReporterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: String::new(),
            token_url: String::new(),
            client_id: String::new(),
            client_secret: String::new(),
            harness_name: String::new(),
            interval_secs: DEFAULT_REPORTER_INTERVAL_SECS,
            timeout_secs: DEFAULT_REPORTER_TIMEOUT_SECS,
            token_skew_secs: DEFAULT_TOKEN_SKEW_SECS,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
        }
    }
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            transport: McpTransport::Stdio,
            command: String::new(),
            args: Vec::new(),
            required: false,
            connect_timeout_secs: DEFAULT_MCP_CONNECT_TIMEOUT_SECS,
            call_timeout_secs: DEFAULT_MCP_CALL_TIMEOUT_SECS,
            env: BTreeMap::new(),
        }
    }
}

impl Config {
    /// First 12 lowercase hex characters of the canonical config hash.
    pub fn fingerprint(&self) -> Result<String, Error> {
        Ok(sha256_12(self.canonical_json()?.as_bytes()))
    }

    fn canonical_json(&self) -> Result<String, Error> {
        let mut cfg = self.clone();
        cfg.llm.api_key.clear();
        cfg.reporter.client_secret.clear();
        cfg.service.token.clear();
        for server in cfg.mcp.servers.values_mut() {
            for value in server.env.values_mut() {
                value.clear();
            }
        }
        // `to_value` lands in a `BTreeMap` (no `preserve_order`), so object keys
        // are already in UTF-8 order. Struct field order is not.
        let value = serde_json::to_value(&cfg)
            .map_err(|err| Error::Config(format!("config fingerprint: {err}")))?;
        serde_json::to_string(&value)
            .map_err(|err| Error::Config(format!("config fingerprint: {err}")))
    }
}

/// Load TOML config, then apply `AGENTSMITH_*` overrides.
///
/// Search order: `explicit`, `AGENTSMITH_CONFIG`, `./agentsmith.toml`,
/// `$XDG_CONFIG_HOME/agentsmith/config.toml` (default `~/.config/agentsmith/config.toml`),
/// then `/etc/agentsmith/config.toml`. A missing file uses built-in defaults.
pub fn load(explicit: Option<&Path>) -> Result<Config, Error> {
    let env = std::env::vars().collect::<BTreeMap<_, _>>();
    let cwd = std::env::current_dir()?;
    load_with(explicit, &env, &cwd, Path::new(ETC_CONFIG))
}

fn load_with(
    explicit: Option<&Path>,
    env: &BTreeMap<String, String>,
    cwd: &Path,
    etc_config: &Path,
) -> Result<Config, Error> {
    match resolve_config_path(explicit, env, cwd, etc_config)? {
        Some(path) => {
            let text = std::fs::read_to_string(&path)
                .map_err(|err| Error::Config(format!("reading {}: {err}", path.display())))?;
            from_toml_str(&text, &path.display().to_string(), env, cwd)
        }
        None => from_toml_str("", "config", env, cwd),
    }
}

fn from_toml_str(
    text: &str,
    origin: &str,
    env: &BTreeMap<String, String>,
    cwd: &Path,
) -> Result<Config, Error> {
    let cfg = if text.trim().is_empty() {
        Config::default()
    } else {
        toml::from_str(text).map_err(|err| Error::Config(format!("{origin}: {err}")))?
    };
    finish(cfg, env, cwd)
}

fn finish(mut cfg: Config, env: &BTreeMap<String, String>, cwd: &Path) -> Result<Config, Error> {
    apply_env(&mut cfg, env)?;
    fill_dynamic(&mut cfg, env, cwd);
    // The report interval is the one clamp. Every other ceiling rejects.
    cfg.reporter.interval_secs = cfg
        .reporter
        .interval_secs
        .clamp(MIN_REPORTER_INTERVAL_SECS, MAX_REPORTER_INTERVAL_SECS);
    validate(&cfg)?;
    disable_incomplete_reporter(&mut cfg);
    Ok(cfg)
}

fn apply_env(cfg: &mut Config, env: &BTreeMap<String, String>) -> Result<(), Error> {
    apply_harness_env(cfg, env)?;
    apply_llm_env(cfg, env)?;
    apply_loop_env(cfg, env)?;
    apply_tools_env(cfg, env)?;
    apply_service_env(cfg, env)?;
    apply_reporter_env(cfg, env)?;
    Ok(())
}

fn apply_harness_env(cfg: &mut Config, env: &BTreeMap<String, String>) -> Result<(), Error> {
    overlay_string(&mut cfg.harness.name, env, "AGENTSMITH_HARNESS_NAME");
    if let Some(raw) = env.get("AGENTSMITH_HARNESS_LOG_LEVEL") {
        cfg.harness.log_level = parse_log_level(raw)?;
    }
    overlay_path(
        &mut cfg.harness.data_dir,
        env,
        "AGENTSMITH_HARNESS_DATA_DIR",
    );
    overlay_path(&mut cfg.harness.workdir, env, "AGENTSMITH_HARNESS_WORKDIR");
    Ok(())
}

fn apply_llm_env(cfg: &mut Config, env: &BTreeMap<String, String>) -> Result<(), Error> {
    overlay_string(&mut cfg.llm.base_url, env, "AGENTSMITH_LLM_BASE_URL");
    overlay_string(&mut cfg.llm.model, env, "AGENTSMITH_LLM_MODEL");
    overlay_string(&mut cfg.llm.api_key, env, "AGENTSMITH_LLM_API_KEY");
    overlay_u64(
        &mut cfg.llm.timeout_secs,
        env,
        "AGENTSMITH_LLM_TIMEOUT_SECS",
    )?;
    overlay_u64(
        &mut cfg.llm.max_output_tokens,
        env,
        "AGENTSMITH_LLM_MAX_OUTPUT_TOKENS",
    )?;
    if let Some(raw) = env.get("AGENTSMITH_LLM_TEMPERATURE") {
        cfg.llm.temperature = Some(parse_f32("AGENTSMITH_LLM_TEMPERATURE", raw)?);
    }
    overlay_u64(&mut cfg.llm.retries, env, "AGENTSMITH_LLM_RETRIES")?;
    Ok(())
}

fn apply_loop_env(cfg: &mut Config, env: &BTreeMap<String, String>) -> Result<(), Error> {
    overlay_u64(
        &mut cfg.loop_.react_max_iterations,
        env,
        "AGENTSMITH_LOOP_REACT_MAX_ITERATIONS",
    )?;
    overlay_u64(
        &mut cfg.loop_.step_max_iterations,
        env,
        "AGENTSMITH_LOOP_STEP_MAX_ITERATIONS",
    )?;
    overlay_u64(
        &mut cfg.loop_.max_subagent_depth,
        env,
        "AGENTSMITH_LOOP_MAX_SUBAGENT_DEPTH",
    )?;
    overlay_u64(&mut cfg.loop_.max_fanout, env, "AGENTSMITH_LOOP_MAX_FANOUT")?;
    overlay_u64(
        &mut cfg.loop_.max_live_subagents,
        env,
        "AGENTSMITH_LOOP_MAX_LIVE_SUBAGENTS",
    )?;
    overlay_u64(
        &mut cfg.loop_.spawn_timeout_secs,
        env,
        "AGENTSMITH_LOOP_SPAWN_TIMEOUT_SECS",
    )?;
    if let Some(raw) = env.get("AGENTSMITH_LOOP_FAILURE_POLICY") {
        cfg.loop_.failure_policy = parse_failure_policy(raw)?;
    }
    Ok(())
}

fn apply_tools_env(cfg: &mut Config, env: &BTreeMap<String, String>) -> Result<(), Error> {
    // Absent stays None. A present variable is what forces Some(true/false).
    if let Some(raw) = env.get("AGENTSMITH_TOOLS_SHELL_ENABLED") {
        cfg.tools.shell.enabled = Some(parse_bool("AGENTSMITH_TOOLS_SHELL_ENABLED", raw)?);
    }
    Ok(())
}

fn apply_service_env(cfg: &mut Config, env: &BTreeMap<String, String>) -> Result<(), Error> {
    overlay_string(&mut cfg.service.bind, env, "AGENTSMITH_SERVICE_BIND");
    overlay_string(&mut cfg.service.token, env, "AGENTSMITH_SERVICE_TOKEN");
    overlay_u64(
        &mut cfg.service.max_runs,
        env,
        "AGENTSMITH_SERVICE_MAX_RUNS",
    )?;
    Ok(())
}

fn apply_reporter_env(cfg: &mut Config, env: &BTreeMap<String, String>) -> Result<(), Error> {
    if let Some(raw) = env.get("AGENTSMITH_REPORTER_ENABLED") {
        cfg.reporter.enabled = parse_bool("AGENTSMITH_REPORTER_ENABLED", raw)?;
    }
    overlay_string(
        &mut cfg.reporter.endpoint,
        env,
        "AGENTSMITH_REPORTER_ENDPOINT",
    );
    overlay_string(
        &mut cfg.reporter.token_url,
        env,
        "AGENTSMITH_REPORTER_TOKEN_URL",
    );
    overlay_string(
        &mut cfg.reporter.client_id,
        env,
        "AGENTSMITH_REPORTER_CLIENT_ID",
    );
    overlay_string(
        &mut cfg.reporter.client_secret,
        env,
        "AGENTSMITH_REPORTER_CLIENT_SECRET",
    );
    overlay_string(
        &mut cfg.reporter.harness_name,
        env,
        "AGENTSMITH_REPORTER_HARNESS_NAME",
    );
    overlay_u64(
        &mut cfg.reporter.interval_secs,
        env,
        "AGENTSMITH_REPORTER_INTERVAL_SECS",
    )?;
    overlay_u64(
        &mut cfg.reporter.timeout_secs,
        env,
        "AGENTSMITH_REPORTER_TIMEOUT_SECS",
    )?;
    Ok(())
}

fn fill_dynamic(cfg: &mut Config, env: &BTreeMap<String, String>, cwd: &Path) {
    if cfg.harness.name.is_empty() {
        cfg.harness.name = hostname();
    }
    if cfg.harness.data_dir.as_os_str().is_empty() {
        cfg.harness.data_dir = default_data_dir(env);
    }
    if cfg.harness.workdir.as_os_str().is_empty() {
        cfg.harness.workdir = cwd.to_path_buf();
    }
    if cfg.reporter.harness_name.is_empty() {
        cfg.reporter.harness_name = cfg.harness.name.clone();
    }
}

fn disable_incomplete_reporter(cfg: &mut Config) {
    if !cfg.reporter.enabled {
        return;
    }
    let incomplete = cfg.reporter.endpoint.is_empty()
        || cfg.reporter.token_url.is_empty()
        || cfg.reporter.client_id.is_empty()
        || cfg.reporter.client_secret.is_empty();
    if incomplete {
        cfg.reporter.enabled = false;
        eprintln!(
            "warning: reporter.enabled is set but endpoint, token_url, client_id, or client_secret is empty; reporter stays disabled"
        );
    }
}

fn validate(cfg: &Config) -> Result<(), Error> {
    validate_llm(cfg)?;
    validate_loop(cfg)?;
    validate_tools(cfg)?;
    validate_service(cfg)?;
    validate_reporter(cfg)?;
    validate_host(cfg)?;
    validate_mcp(cfg)?;
    Ok(())
}

fn validate_llm(cfg: &Config) -> Result<(), Error> {
    check_url("llm.base_url", &cfg.llm.base_url, false)?;
    if let Some(temperature) = cfg.llm.temperature {
        if !temperature.is_finite() {
            return Err(Error::Config(
                "llm.temperature must be a finite number".into(),
            ));
        }
    }
    ceiling(
        "llm.timeout_secs",
        cfg.llm.timeout_secs,
        CEILING_LLM_TIMEOUT_SECS,
    )?;
    ceiling("llm.retries", cfg.llm.retries, CEILING_LLM_RETRIES)?;
    Ok(())
}

fn validate_loop(cfg: &Config) -> Result<(), Error> {
    ceiling(
        "loop.react_max_iterations",
        cfg.loop_.react_max_iterations,
        CEILING_REACT_MAX_ITERATIONS,
    )?;
    ceiling(
        "loop.step_max_iterations",
        cfg.loop_.step_max_iterations,
        CEILING_STEP_MAX_ITERATIONS,
    )?;
    ceiling(
        "loop.planner_attempts",
        cfg.loop_.planner_attempts,
        CEILING_PLANNER_ATTEMPTS,
    )?;
    ceiling(
        "loop.max_consecutive_tool_errors",
        cfg.loop_.max_consecutive_tool_errors,
        CEILING_MAX_CONSECUTIVE_TOOL_ERRORS,
    )?;
    ceiling(
        "loop.max_subagent_depth",
        cfg.loop_.max_subagent_depth,
        CEILING_MAX_SUBAGENT_DEPTH,
    )?;
    ceiling("loop.max_fanout", cfg.loop_.max_fanout, CEILING_MAX_FANOUT)?;
    ceiling(
        "loop.max_live_subagents",
        cfg.loop_.max_live_subagents,
        CEILING_MAX_LIVE_SUBAGENTS,
    )?;
    ceiling(
        "loop.spawn_timeout_secs",
        cfg.loop_.spawn_timeout_secs,
        CEILING_SPAWN_TIMEOUT_SECS,
    )?;
    Ok(())
}

fn validate_tools(cfg: &Config) -> Result<(), Error> {
    ceiling(
        "tools.max_output_bytes",
        cfg.tools.max_output_bytes,
        CEILING_TOOL_MAX_OUTPUT_BYTES,
    )?;
    ceiling(
        "tools.shell.timeout_secs",
        cfg.tools.shell.timeout_secs,
        CEILING_SHELL_TIMEOUT_SECS,
    )?;
    ceiling(
        "tools.shell.max_timeout_secs",
        cfg.tools.shell.max_timeout_secs,
        CEILING_SHELL_MAX_TIMEOUT_SECS,
    )?;
    if cfg.tools.shell.timeout_secs > cfg.tools.shell.max_timeout_secs {
        return Err(Error::Config(format!(
            "tools.shell.timeout_secs {} exceeds tools.shell.max_timeout_secs {}",
            cfg.tools.shell.timeout_secs, cfg.tools.shell.max_timeout_secs
        )));
    }
    ceiling(
        "tools.files.max_read_bytes",
        cfg.tools.files.max_read_bytes,
        CEILING_MAX_READ_BYTES,
    )?;
    ceiling(
        "tools.files.max_write_bytes",
        cfg.tools.files.max_write_bytes,
        CEILING_MAX_WRITE_BYTES,
    )?;
    Ok(())
}

fn validate_service(cfg: &Config) -> Result<(), Error> {
    check_bind(&cfg.service.bind)?;
    ceiling("service.max_runs", cfg.service.max_runs, CEILING_MAX_RUNS)?;
    ceiling(
        "service.shutdown_grace_secs",
        cfg.service.shutdown_grace_secs,
        CEILING_SHUTDOWN_GRACE_SECS,
    )?;
    // Active runs must fit in the AHSP `runs` array (maxItems 64).
    let slots = cfg
        .service
        .max_runs
        .saturating_mul(1u64.saturating_add(cfg.loop_.max_live_subagents));
    if slots > CEILING_RUN_SLOTS {
        return Err(Error::Config(format!(
            "service.max_runs * (1 + loop.max_live_subagents) is {slots}, ceiling is {CEILING_RUN_SLOTS}"
        )));
    }
    Ok(())
}

fn validate_reporter(cfg: &Config) -> Result<(), Error> {
    check_url("reporter.endpoint", &cfg.reporter.endpoint, true)?;
    check_url("reporter.token_url", &cfg.reporter.token_url, true)?;
    ceiling(
        "reporter.timeout_secs",
        cfg.reporter.timeout_secs,
        CEILING_REPORTER_TIMEOUT_SECS,
    )?;
    ceiling(
        "reporter.max_body_bytes",
        cfg.reporter.max_body_bytes,
        CEILING_MAX_BODY_BYTES,
    )?;
    Ok(())
}

fn validate_host(cfg: &Config) -> Result<(), Error> {
    if cfg.host.watch.len() > CEILING_HOST_WATCH {
        return Err(Error::Config(format!(
            "host.watch has {} entries, ceiling is {CEILING_HOST_WATCH}",
            cfg.host.watch.len()
        )));
    }
    Ok(())
}

const BUILTIN_TOOL_NAMES: &[&str] = &[
    "read_file",
    "write_file",
    "list_dir",
    "shell",
    "finish",
    "spawn_subagents",
];

fn validate_mcp(cfg: &Config) -> Result<(), Error> {
    for (name, server) in &cfg.mcp.servers {
        validate_mcp_server_name(name)?;
        ceiling(
            &format!("mcp.servers.{name}.connect_timeout_secs"),
            server.connect_timeout_secs,
            CEILING_MCP_CONNECT_TIMEOUT_SECS,
        )?;
        ceiling(
            &format!("mcp.servers.{name}.call_timeout_secs"),
            server.call_timeout_secs,
            CEILING_MCP_CALL_TIMEOUT_SECS,
        )?;
    }
    Ok(())
}

fn validate_mcp_server_name(name: &str) -> Result<(), Error> {
    let bytes = name.as_bytes();
    let shaped = bytes.len() <= 32
        && bytes
            .first()
            .is_some_and(|first| first.is_ascii_lowercase())
        && bytes
            .iter()
            .skip(1)
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'_');
    if !shaped || name.contains("__") {
        return Err(Error::Config(format!(
            "mcp server name {name:?} must match ^[a-z][a-z0-9_]{{0,31}}$ and must not contain __"
        )));
    }
    if BUILTIN_TOOL_NAMES.contains(&name) {
        return Err(Error::Config(format!(
            "mcp server name {name:?} must not equal a builtin tool name"
        )));
    }
    Ok(())
}

fn ceiling(field: &str, value: u64, max: u64) -> Result<(), Error> {
    if value > max {
        Err(Error::Config(format!(
            "{field} {value} exceeds ceiling {max}"
        )))
    } else {
        Ok(())
    }
}

fn check_bind(bind: &str) -> Result<(), Error> {
    let loopback = if let Ok(addr) = bind.parse::<SocketAddr>() {
        addr.ip().is_loopback()
    } else if let Some((host, port)) = bind.rsplit_once(':') {
        !port.is_empty() && port.parse::<u16>().is_ok() && host.eq_ignore_ascii_case("localhost")
    } else {
        false
    };
    if loopback {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "service.bind must be loopback (127.0.0.1, ::1, or localhost), got {bind}"
        )))
    }
}

fn check_url(field: &str, raw: &str, allow_empty: bool) -> Result<(), Error> {
    if raw.is_empty() {
        if allow_empty {
            return Ok(());
        }
        return Err(Error::Config(format!("{field} must not be empty")));
    }
    let parsed = Url::parse(raw).map_err(|err| Error::Config(format!("{field}: {err}")))?;
    if parsed.scheme() == "https" {
        return Ok(());
    }
    if parsed.scheme() == "http" && is_loopback_url_host(&parsed) {
        return Ok(());
    }
    Err(Error::Config(format!(
        "{field} must be https unless the host is localhost, 127.0.0.1, or ::1"
    )))
}

fn is_loopback_url_host(url: &Url) -> bool {
    match url.host() {
        Some(url::Host::Domain(domain)) => domain.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(ip)) => ip == Ipv4Addr::LOCALHOST,
        Some(url::Host::Ipv6(ip)) => ip == Ipv6Addr::LOCALHOST,
        None => false,
    }
}

fn resolve_config_path(
    explicit: Option<&Path>,
    env: &BTreeMap<String, String>,
    cwd: &Path,
    etc_config: &Path,
) -> Result<Option<PathBuf>, Error> {
    if let Some(path) = explicit {
        return require_file(&anchored(cwd, path)).map(Some);
    }
    if let Some(raw) = env
        .get("AGENTSMITH_CONFIG")
        .filter(|value| !value.is_empty())
    {
        return require_file(&anchored(cwd, Path::new(raw))).map(Some);
    }
    let local = cwd.join("agentsmith.toml");
    if local.is_file() {
        return Ok(Some(local));
    }
    if let Some(xdg) = xdg_config_file(env) {
        if xdg.is_file() {
            return Ok(Some(xdg));
        }
    }
    if etc_config.is_file() {
        return Ok(Some(etc_config.to_path_buf()));
    }
    Ok(None)
}

fn require_file(path: &Path) -> Result<PathBuf, Error> {
    if path.is_file() {
        Ok(path.to_path_buf())
    } else {
        Err(Error::Config(format!(
            "config file not found: {}",
            path.display()
        )))
    }
}

fn anchored(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

fn xdg_config_file(env: &BTreeMap<String, String>) -> Option<PathBuf> {
    if let Some(xdg) = env.get("XDG_CONFIG_HOME").filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(xdg).join("agentsmith").join("config.toml"));
    }
    let home = env.get("HOME").filter(|value| !value.is_empty())?;
    Some(
        PathBuf::from(home)
            .join(".config")
            .join("agentsmith")
            .join("config.toml"),
    )
}

fn default_data_dir(env: &BTreeMap<String, String>) -> PathBuf {
    if let Some(xdg) = env.get("XDG_DATA_HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(xdg).join("agentsmith");
    }
    if let Some(home) = env.get("HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(home).join(".local/share/agentsmith");
    }
    PathBuf::from(".local/share/agentsmith")
}

fn hostname() -> String {
    let raw = gethostname::gethostname();
    let name = raw.to_string_lossy();
    let name = name.trim();
    if name.is_empty() {
        "unknown".to_string()
    } else {
        name.to_string()
    }
}

fn overlay_string(slot: &mut String, env: &BTreeMap<String, String>, key: &str) {
    if let Some(value) = env.get(key) {
        *slot = value.clone();
    }
}

fn overlay_path(slot: &mut PathBuf, env: &BTreeMap<String, String>, key: &str) {
    if let Some(value) = env.get(key) {
        *slot = PathBuf::from(value);
    }
}

fn overlay_u64(slot: &mut u64, env: &BTreeMap<String, String>, key: &str) -> Result<(), Error> {
    if let Some(value) = env.get(key) {
        *slot = value
            .parse::<u64>()
            .map_err(|_| Error::Config(format!("{key} must be an integer, got {value}")))?;
    }
    Ok(())
}

fn parse_f32(key: &str, raw: &str) -> Result<f32, Error> {
    let value = raw
        .parse::<f32>()
        .map_err(|_| Error::Config(format!("{key} must be a number, got {raw}")))?;
    if !value.is_finite() {
        return Err(Error::Config(format!("{key} must be a finite number")));
    }
    Ok(value)
}

fn parse_bool(key: &str, raw: &str) -> Result<bool, Error> {
    match raw {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(Error::Config(format!("{key} must be true or false"))),
    }
}

fn parse_log_level(raw: &str) -> Result<LogLevel, Error> {
    match raw {
        "trace" => Ok(LogLevel::Trace),
        "debug" => Ok(LogLevel::Debug),
        "info" => Ok(LogLevel::Info),
        "warn" => Ok(LogLevel::Warn),
        "error" => Ok(LogLevel::Error),
        _ => Err(Error::Config(format!(
            "harness.log_level must be trace|debug|info|warn|error, got {raw}"
        ))),
    }
}

fn parse_failure_policy(raw: &str) -> Result<FailurePolicy, Error> {
    match raw {
        "continue" => Ok(FailurePolicy::Continue),
        "fail_fast" => Ok(FailurePolicy::FailFast),
        _ => Err(Error::Config(format!(
            "loop.failure_policy must be continue|fail_fast, got {raw}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_env() -> BTreeMap<String, String> {
        BTreeMap::from([("HOME".to_string(), "/home/tester".to_string())])
    }

    fn workdir() -> PathBuf {
        PathBuf::from("/work")
    }

    fn load_toml(text: &str) -> Result<Config, Error> {
        load_toml_with(text, &test_env())
    }

    fn load_toml_with(text: &str, env: &BTreeMap<String, String>) -> Result<Config, Error> {
        from_toml_str(text, "config", env, &workdir())
    }

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("agentsmith-cfg-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn defaults_load_without_model_or_api_key() {
        let cfg = load_toml("").unwrap();
        assert!(cfg.llm.model.is_empty());
        assert!(cfg.llm.api_key.is_empty());
        assert!(cfg.llm.temperature.is_none());
        assert_eq!(cfg.llm.base_url, DEFAULT_BASE_URL);
        assert_eq!(cfg.llm.timeout_secs, 120);
        assert_eq!(cfg.llm.max_output_tokens, 4096);
        assert_eq!(cfg.llm.retries, 2);
        assert_eq!(cfg.loop_.react_max_iterations, 25);
        assert_eq!(cfg.loop_.step_max_iterations, 8);
        assert_eq!(cfg.loop_.planner_attempts, 2);
        assert_eq!(cfg.loop_.max_fanout, 8);
        assert_eq!(cfg.loop_.max_live_subagents, 16);
        assert_eq!(cfg.loop_.failure_policy, FailurePolicy::Continue);
        assert_eq!(cfg.tools.shell.enabled, None);
        assert_eq!(cfg.tools.shell.timeout_secs, 30);
        assert_eq!(cfg.tools.max_output_bytes, 32_768);
        assert_eq!(cfg.tools.files.max_read_bytes, 65_536);
        assert_eq!(cfg.tools.files.max_write_bytes, 262_144);
        assert_eq!(cfg.service.bind, "127.0.0.1:7433");
        assert_eq!(cfg.service.max_runs, 2);
        assert!(!cfg.reporter.enabled);
        assert_eq!(cfg.reporter.interval_secs, 30);
        assert_eq!(cfg.reporter.max_body_bytes, 262_144);
        assert_eq!(cfg.harness.log_level, LogLevel::Info);
        assert_eq!(cfg.harness.workdir, workdir());
        assert_eq!(
            cfg.harness.data_dir,
            PathBuf::from("/home/tester/.local/share/agentsmith")
        );
        assert!(!cfg.harness.name.is_empty());
        assert_eq!(cfg.reporter.harness_name, cfg.harness.name);
        let fp = cfg.fingerprint().unwrap();
        assert_eq!(fp.len(), 12);
        assert!(fp.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')));
        assert!(!cfg.canonical_json().unwrap().contains("temperature"));
    }

    #[test]
    fn secrets_do_not_change_the_fingerprint() {
        let base = r#"
            [llm]
            api_key = "sentinel-api-key-a"
            [service]
            token = "sentinel-service-token-a"
            [reporter]
            client_secret = "sentinel-client-secret-a"
            [mcp.servers.fs]
            command = "npx"
            [mcp.servers.fs.env]
            NPM_CONFIG_CACHE = "sentinel-mcp-env-a"
        "#;
        let other = r#"
            [service]
            token = "sentinel-service-token-b"
            [llm]
            api_key = "sentinel-api-key-b"
            [mcp.servers.fs.env]
            NPM_CONFIG_CACHE = "sentinel-mcp-env-b"
            [mcp.servers.fs]
            command = "npx"
            [reporter]
            client_secret = "sentinel-client-secret-b"
        "#;
        let left = load_toml(base).unwrap();
        let right = load_toml(other).unwrap();
        assert_eq!(
            left.canonical_json().unwrap(),
            right.canonical_json().unwrap()
        );
        assert_eq!(left.fingerprint().unwrap(), right.fingerprint().unwrap());
        let json = left.canonical_json().unwrap();
        assert!(!json.contains("sentinel-api-key"));
        assert!(!json.contains("sentinel-client-secret"));
        assert!(!json.contains("sentinel-service-token"));
        assert!(!json.contains("sentinel-mcp-env"));
        assert!(json.contains("\"NPM_CONFIG_CACHE\":\"\""));
    }

    #[test]
    fn shell_enabled_null_and_false_hash_differently() {
        let absent = load_toml("").unwrap();
        let disabled = load_toml("[tools.shell]\nenabled = false\n").unwrap();
        assert_eq!(absent.tools.shell.enabled, None);
        assert_eq!(disabled.tools.shell.enabled, Some(false));
        let absent_json = absent.canonical_json().unwrap();
        let disabled_json = disabled.canonical_json().unwrap();
        assert!(absent_json.contains("\"enabled\":null"), "{absent_json}");
        assert!(
            disabled_json.contains("\"enabled\":false"),
            "{disabled_json}"
        );
        assert_ne!(
            absent.fingerprint().unwrap(),
            disabled.fingerprint().unwrap()
        );
    }

    #[test]
    fn toml_key_order_does_not_change_the_fingerprint() {
        let first = r#"
            [llm]
            model = "m"
            base_url = "https://api.openai.com/v1"
            [service]
            token = "t"
            bind = "127.0.0.1:1"
            [mcp.servers.fs.env]
            B = "1"
            A = "2"
            [mcp.servers.fs]
            command = "npx"
            transport = "stdio"
        "#;
        let second = r#"
            [mcp.servers.fs]
            transport = "stdio"
            command = "npx"
            [mcp.servers.fs.env]
            A = "2"
            B = "1"
            [service]
            bind = "127.0.0.1:1"
            token = "t"
            [llm]
            base_url = "https://api.openai.com/v1"
            model = "m"
        "#;
        let left = load_toml(first).unwrap();
        let right = load_toml(second).unwrap();
        assert_eq!(
            left.canonical_json().unwrap(),
            right.canonical_json().unwrap()
        );
        assert_eq!(left.fingerprint().unwrap(), right.fingerprint().unwrap());
    }

    #[test]
    fn watch_array_order_is_part_of_the_fingerprint() {
        let forward = "[[host.watch]]\nname = \"a\"\nbinary = \"a\"\n[[host.watch]]\nname = \"b\"\nbinary = \"b\"\n";
        let reverse = "[[host.watch]]\nname = \"b\"\nbinary = \"b\"\n[[host.watch]]\nname = \"a\"\nbinary = \"a\"\n";
        assert_ne!(
            load_toml(forward).unwrap().fingerprint().unwrap(),
            load_toml(reverse).unwrap().fingerprint().unwrap()
        );
    }

    #[test]
    fn ceilings_bind_and_transport_reject() {
        assert!(load_toml("[loop]\nplanner_attempts = 2\nmax_fanout = 8\n").is_ok());
        assert!(load_toml("[loop]\nplanner_attempts = 3\n").is_err());
        assert!(load_toml("[loop]\nmax_fanout = 9\n").is_err());
        assert!(load_toml("[reporter]\nmax_body_bytes = 262144\n").is_ok());
        assert!(load_toml("[reporter]\nmax_body_bytes = 262145\n").is_err());
        assert!(load_toml("[service]\nbind = \"0.0.0.0:7433\"\n").is_err());
        assert!(load_toml("[service]\nbind = \"192.168.1.5:80\"\n").is_err());
        assert!(load_toml("[service]\nbind = \"localhost:7433\"\n").is_ok());
        assert!(load_toml("[service]\nbind = \"[::1]:7433\"\n").is_ok());
        assert!(load_toml("[mcp.servers.fs]\ntransport = \"http\"\n").is_err());
        assert!(load_toml("[mcp.servers.fs]\ntransport = \"stdio\"\n").is_ok());
        assert!(load_toml("harness.log_level = \"verbose\"\n").is_err());
        assert!(load_toml("loop.failure_policy = \"retry\"\n").is_err());
        assert!(
            load_toml("harness.log_level = \"debug\"\nloop.failure_policy = \"fail_fast\"\n")
                .is_ok()
        );
    }

    #[test]
    fn run_slot_product_is_rejected_above_64() {
        assert!(load_toml("[service]\nmax_runs = 8\n[loop]\nmax_live_subagents = 7\n").is_ok());
        let err = load_toml("[service]\nmax_runs = 8\n[loop]\nmax_live_subagents = 8\n")
            .expect_err("8 * 9 = 72");
        assert!(err.to_string().contains("64"), "{err}");
        assert!(load_toml("[service]\nmax_runs = 9\n").is_err());
    }

    #[test]
    fn host_watch_rejects_more_than_32() {
        let mut text = String::new();
        for i in 0..33 {
            text.push_str(&format!(
                "[[host.watch]]\nname = \"n{i}\"\nbinary = \"b{i}\"\n"
            ));
        }
        assert!(load_toml(&text).is_err());
        text.clear();
        for i in 0..32 {
            text.push_str(&format!(
                "[[host.watch]]\nname = \"n{i}\"\nbinary = \"b{i}\"\n"
            ));
        }
        assert!(load_toml(&text).is_ok());
    }

    #[test]
    fn env_override_beats_the_file_and_unknown_vars_are_ignored() {
        let mut env = test_env();
        env.insert("AGENTSMITH_LLM_MODEL".into(), "from-env".into());
        env.insert("AGENTSMITH_TOOLS_SHELL_ENABLED".into(), "false".into());
        env.insert("AGENTSMITH_LOOP_MAX_FANOUT".into(), "4".into());
        env.insert("AGENTSMITH_NOT_A_REAL_KEY".into(), "1".into());
        env.insert("AGENTSMITH_LOOP_PLANNER_ATTEMPTS".into(), "1".into());
        let cfg = load_toml_with(
            r#"
            [llm]
            model = "from-file"
            [loop]
            planner_attempts = 2
            max_fanout = 8
            "#,
            &env,
        )
        .unwrap();
        assert_eq!(cfg.llm.model, "from-env");
        assert_eq!(cfg.tools.shell.enabled, Some(false));
        assert_eq!(cfg.loop_.max_fanout, 4);
        assert_eq!(cfg.loop_.planner_attempts, 2);
    }

    #[test]
    fn env_shell_enabled_true_sets_some() {
        let mut env = test_env();
        env.insert("AGENTSMITH_TOOLS_SHELL_ENABLED".into(), "true".into());
        let cfg = load_toml_with("", &env).unwrap();
        assert_eq!(cfg.tools.shell.enabled, Some(true));
    }

    #[test]
    fn env_above_a_ceiling_fails_load() {
        let mut env = test_env();
        env.insert("AGENTSMITH_LOOP_MAX_FANOUT".into(), "9".into());
        assert!(load_toml_with("[loop]\nmax_fanout = 8\n", &env).is_err());
    }

    #[test]
    fn reporter_interval_is_clamped() {
        assert_eq!(
            load_toml("[reporter]\ninterval_secs = 1\n")
                .unwrap()
                .reporter
                .interval_secs,
            10
        );
        assert_eq!(
            load_toml("[reporter]\ninterval_secs = 9999\n")
                .unwrap()
                .reporter
                .interval_secs,
            300
        );
        assert_eq!(
            load_toml("[reporter]\ninterval_secs = 30\n")
                .unwrap()
                .reporter
                .interval_secs,
            30
        );
    }

    #[test]
    fn incomplete_reporter_stays_disabled() {
        let cfg = load_toml(
            r#"
            [reporter]
            enabled = true
            endpoint = "https://control.example/state"
            token_url = "https://auth.example/token"
            client_id = "id"
            "#,
        )
        .unwrap();
        assert!(!cfg.reporter.enabled);

        let cfg = load_toml(
            r#"
            [harness]
            name = "box"
            [reporter]
            enabled = true
            endpoint = "https://control.example/state"
            token_url = "https://auth.example/token"
            client_id = "id"
            client_secret = "sec"
            "#,
        )
        .unwrap();
        assert!(cfg.reporter.enabled);
        assert_eq!(cfg.reporter.harness_name, "box");

        let cfg = load_toml(
            r#"
            [harness]
            name = "box"
            [reporter]
            harness_name = "other"
            "#,
        )
        .unwrap();
        assert_eq!(cfg.reporter.harness_name, "other");
    }

    #[test]
    fn urls_must_be_https_or_loopback() {
        assert!(load_toml("[llm]\nbase_url = \"http://example.com/v1\"\n").is_err());
        assert!(load_toml("[llm]\nbase_url = \"http://127.0.0.1:9/v1\"\n").is_ok());
        assert!(load_toml("[llm]\nbase_url = \"http://localhost:9/v1\"\n").is_ok());
        assert!(load_toml("[llm]\nbase_url = \"http://[::1]:9/v1\"\n").is_ok());
        assert!(load_toml("[reporter]\nendpoint = \"http://10.0.0.1/x\"\n").is_err());
        assert!(load_toml("[reporter]\nendpoint = \"\"\ntoken_url = \"\"\n").is_ok());
        assert!(load_toml("[llm]\ntemperature = 0.2\n")
            .unwrap()
            .llm
            .temperature
            .is_some());
    }

    #[test]
    fn data_dir_honors_xdg_data_home() {
        let mut env = test_env();
        env.insert("XDG_DATA_HOME".into(), "/var/data".into());
        let cfg = load_toml_with("", &env).unwrap();
        assert_eq!(cfg.harness.data_dir, PathBuf::from("/var/data/agentsmith"));
    }

    #[test]
    fn search_order_explicit_then_env_then_cwd_then_xdg_then_etc() {
        let scratch = Scratch::new();
        let cwd = scratch.0.join("cwd");
        let xdg = scratch.0.join("xdg");
        let home = scratch.0.join("home");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(xdg.join("agentsmith")).unwrap();
        std::fs::create_dir_all(home.join(".config/agentsmith")).unwrap();
        std::fs::write(cwd.join("agentsmith.toml"), "harness.name = \"from-cwd\"\n").unwrap();
        std::fs::write(
            xdg.join("agentsmith/config.toml"),
            "harness.name = \"from-xdg\"\n",
        )
        .unwrap();
        std::fs::write(
            home.join(".config/agentsmith/config.toml"),
            "harness.name = \"from-home\"\n",
        )
        .unwrap();
        let etc = scratch.0.join("etc.toml");
        std::fs::write(&etc, "harness.name = \"from-etc\"\n").unwrap();
        let env_path = scratch.0.join("env.toml");
        std::fs::write(&env_path, "harness.name = \"from-env-path\"\n").unwrap();
        let explicit = scratch.0.join("explicit.toml");
        std::fs::write(&explicit, "harness.name = \"from-explicit\"\n").unwrap();

        let mut env = test_env();
        env.insert("XDG_CONFIG_HOME".into(), xdg.to_string_lossy().into_owned());
        env.insert("HOME".into(), home.to_string_lossy().into_owned());

        let cfg = load_with(None, &env, &cwd, &etc).unwrap();
        assert_eq!(cfg.harness.name, "from-cwd");

        let empty_cwd = scratch.0.join("empty-cwd");
        std::fs::create_dir_all(&empty_cwd).unwrap();
        let cfg = load_with(None, &env, &empty_cwd, &etc).unwrap();
        assert_eq!(cfg.harness.name, "from-xdg");

        env.remove("XDG_CONFIG_HOME");
        let cfg = load_with(None, &env, &empty_cwd, &etc).unwrap();
        assert_eq!(cfg.harness.name, "from-home");

        let no_home = scratch.0.join("no-home");
        std::fs::create_dir_all(&no_home).unwrap();
        env.remove("HOME");
        let cfg = load_with(None, &env, &no_home, &etc).unwrap();
        assert_eq!(cfg.harness.name, "from-etc");

        let missing_etc = scratch.0.join("missing.toml");
        let cfg = load_with(None, &env, &no_home, &missing_etc).unwrap();
        assert_ne!(cfg.harness.name, "from-etc");
        assert!(!cfg.harness.name.is_empty());

        env.insert(
            "AGENTSMITH_CONFIG".into(),
            env_path.to_string_lossy().into_owned(),
        );
        let cfg = load_with(None, &env, &cwd, &etc).unwrap();
        assert_eq!(cfg.harness.name, "from-env-path");

        let cfg = load_with(Some(&explicit), &env, &cwd, &etc).unwrap();
        assert_eq!(cfg.harness.name, "from-explicit");

        let relative_cwd = scratch.0.join("rel");
        std::fs::create_dir_all(&relative_cwd).unwrap();
        std::fs::write(
            relative_cwd.join("custom.toml"),
            "harness.name = \"from-relative\"\n",
        )
        .unwrap();
        let cfg = load_with(Some(Path::new("custom.toml")), &env, &relative_cwd, &etc).unwrap();
        assert_eq!(cfg.harness.name, "from-relative");

        let err = load_with(
            Some(scratch.0.join("nope.toml").as_path()),
            &env,
            &cwd,
            &etc,
        )
        .expect_err("missing explicit");
        assert!(err.to_string().contains("not found"), "{err}");

        env.insert(
            "AGENTSMITH_CONFIG".into(),
            "/no/such/agentsmith.toml".into(),
        );
        assert!(load_with(None, &env, &empty_cwd, &etc).is_err());
    }

    #[test]
    fn non_finite_temperature_is_rejected() {
        for raw in ["nan", "inf", "-inf", "1e39"] {
            let err = load_toml(&format!("[llm]\ntemperature = {raw}\n")).expect_err(raw);
            assert!(
                err.to_string()
                    .contains("llm.temperature must be a finite number"),
                "{raw}: {err}"
            );
        }
        assert_eq!(
            load_toml("[llm]\ntemperature = 0.2\n")
                .unwrap()
                .llm
                .temperature,
            Some(0.2)
        );
    }

    #[test]
    fn mcp_server_names_are_restricted() {
        for name in [
            "foo__bar",
            "shell",
            "FS",
            "9abc",
            "foo-bar",
            "read_file",
            "write_file",
            "list_dir",
            "finish",
            "spawn_subagents",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ] {
            let err =
                load_toml(&format!("[mcp.servers.{name}]\ncommand = \"npx\"\n")).expect_err(name);
            assert!(err.to_string().contains("mcp server name"), "{name}: {err}");
        }
        assert!(load_toml("[mcp.servers.fs]\ncommand = \"npx\"\n").is_ok());
        let ok = "a".repeat(32);
        assert!(load_toml(&format!("[mcp.servers.{ok}]\ncommand = \"npx\"\n")).is_ok());
    }

    #[test]
    fn non_stdio_transport_names_the_v1_limit() {
        for raw in ["http", "sse"] {
            let err = load_toml(&format!("[mcp.servers.fs]\ntransport = \"{raw}\"\n")).unwrap_err();
            assert!(
                err.to_string()
                    .contains("streamable HTTP MCP is not supported in v1"),
                "{raw}: {err}"
            );
        }
    }

    #[test]
    fn shell_timeout_cannot_exceed_its_max() {
        let err =
            load_toml("[tools.shell]\ntimeout_secs = 120\nmax_timeout_secs = 30\n").unwrap_err();
        assert!(err.to_string().contains("timeout_secs"), "{err}");
        assert!(err.to_string().contains("max_timeout_secs"), "{err}");
        assert!(load_toml("[tools.shell]\ntimeout_secs = 30\nmax_timeout_secs = 30\n").is_ok());
        assert!(load_toml("[tools.shell]\ntimeout_secs = 120\nmax_timeout_secs = 120\n").is_ok());
    }

    #[test]
    fn partial_tables_keep_section_defaults() {
        let cfg = load_toml("[llm]\nmodel = \"only\"\n").unwrap();
        assert_eq!(cfg.llm.model, "only");
        assert_eq!(cfg.llm.timeout_secs, DEFAULT_LLM_TIMEOUT_SECS);
        assert_eq!(cfg.llm.base_url, DEFAULT_BASE_URL);
        assert_eq!(cfg.llm.retries, DEFAULT_LLM_RETRIES);
        assert_eq!(cfg.llm.max_output_tokens, DEFAULT_MAX_OUTPUT_TOKENS);
        assert_eq!(cfg.tools.shell.enabled, None);
        assert_eq!(cfg.tools.shell.timeout_secs, DEFAULT_SHELL_TIMEOUT_SECS);

        let cfg = load_toml("[loop]\nmax_fanout = 4\n").unwrap();
        assert_eq!(cfg.loop_.max_fanout, 4);
        assert_eq!(cfg.loop_.react_max_iterations, DEFAULT_REACT_MAX_ITERATIONS);

        let cfg = load_toml("[mcp.servers.fs]\ncommand = \"npx\"\n").unwrap();
        let server = &cfg.mcp.servers["fs"];
        assert_eq!(server.command, "npx");
        assert_eq!(
            server.connect_timeout_secs,
            DEFAULT_MCP_CONNECT_TIMEOUT_SECS
        );
        assert_eq!(server.call_timeout_secs, DEFAULT_MCP_CALL_TIMEOUT_SECS);
        assert_eq!(server.transport, McpTransport::Stdio);

        let cfg = load_toml("[[host.watch]]\nname = \"n\"\nbinary = \"b\"\n").unwrap();
        assert!(!cfg.host.watch[0].match_node);
    }

    #[cfg(unix)]
    #[test]
    fn fingerprint_of_non_utf8_workdir_is_a_config_error() {
        use std::os::unix::ffi::OsStringExt;

        let mut cfg = load_toml("").unwrap();
        cfg.harness.workdir = PathBuf::from(std::ffi::OsString::from_vec(vec![0xff]));
        let err = cfg.fingerprint().expect_err("non-utf8 workdir");
        match err {
            Error::Config(message) => assert!(message.contains("config fingerprint"), "{message}"),
            other => panic!("expected config error, got {other}"),
        }
    }
}
