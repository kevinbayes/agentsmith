use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use agentsmith_core::config::McpServerConfig;
use agentsmith_core::{Error, ToolErrorKind};
use agentsmith_mcp::{McpSupervisor, McpSupervisorConfig};
use agentsmith_tools::{Tool, ToolCtx, ToolDefinition, ToolOutput, ToolRegistry};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

struct Fixed {
    name: String,
    body: String,
}

#[async_trait::async_trait]
impl Tool for Fixed {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.clone(),
            description: "builtin".into(),
            parameters: json!({ "type": "object" }),
        }
    }

    async fn call(&self, _ctx: ToolCtx, _arguments: Value) -> Result<ToolOutput, Error> {
        Ok(ToolOutput {
            content: self.body.clone(),
        })
    }
}

struct Session {
    dir: PathBuf,
    log: PathBuf,
    registry: ToolRegistry,
    supervisor: Arc<McpSupervisor>,
}

impl Session {
    async fn start(mode: &str, notify: bool, call_timeout_secs: u64) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "agentsmith-mcp-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("peer.log");
        let mut env = BTreeMap::new();
        env.insert("MCP_PEER_LOG".into(), log.display().to_string());
        if notify {
            env.insert("MCP_PEER_NOTIFY".into(), "1".into());
        }
        let args = if mode == "ok" {
            Vec::new()
        } else {
            vec![mode.to_string()]
        };
        let registry = ToolRegistry::new();
        let mut servers = BTreeMap::new();
        servers.insert(
            "fs".into(),
            McpServerConfig {
                command: peer_bin(),
                args,
                connect_timeout_secs: 5,
                call_timeout_secs,
                env,
                ..McpServerConfig::default()
            },
        );
        let supervisor = McpSupervisor::new(McpSupervisorConfig {
            registry: registry.clone(),
            servers,
            max_output_bytes: 32 * 1024,
            shutdown_grace: Duration::from_secs(15),
            cancel: CancellationToken::new(),
        });
        supervisor.poll().await.unwrap();
        Self {
            dir,
            log,
            registry,
            supervisor,
        }
    }

    fn log_text(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_default()
    }

    async fn call(&self, tool: &str, arguments: Value) -> Result<ToolOutput, Error> {
        let ctx = ToolCtx {
            run_id: agentsmith_core::id::new_id(),
            workdir: self.dir.clone(),
            cancel: CancellationToken::new(),
        };
        self.registry
            .get(&format!("fs__{tool}"))
            .unwrap_or_else(|| panic!("missing fs__{tool}"))
            .call(ctx, arguments)
            .await
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn peer_bin() -> String {
    if let Some(path) = option_env!("CARGO_BIN_EXE_mcp_stdio_peer") {
        return path.to_string();
    }
    // Integration-test layout is target/<profile>/deps/<test>. The bin sits beside deps.
    let exe = std::env::current_exe().expect("test executable");
    exe.parent()
        .and_then(|dir| dir.parent())
        .expect("target profile dir")
        .join("mcp-stdio-peer")
        .display()
        .to_string()
}

fn tool_err(err: Error) -> (ToolErrorKind, String) {
    match err {
        Error::Tool { kind, message, .. } => (kind, message),
        other => panic!("expected tool error, got {other}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_and_call_round_trip_namespaces_and_omits_non_text() {
    let session = Session::start("ok", false, 5).await;
    assert_eq!(session.supervisor.tools_skipped("fs"), 3);
    assert!(session.registry.get("fs__echo").is_some());
    assert!(session.registry.get("fs__read_file").is_some());
    assert!(session.registry.get("fs__boom").is_some());
    assert!(session
        .registry
        .names()
        .iter()
        .all(|name| name != "fs__" && !name.contains(' ')));

    let out = session
        .call(
            "echo",
            json!({ "text": "hi", "image": true, "resource": true }),
        )
        .await
        .unwrap();
    assert_eq!(out.content, "hi[image omitted][resource omitted]");

    let read = session.call("read_file", json!({})).await.unwrap();
    assert_eq!(read.content, "mcp-read");

    let (kind, message) = tool_err(session.call("boom", json!({})).await.unwrap_err());
    assert_eq!(kind, ToolErrorKind::Execution);
    assert_eq!(message, "nope");

    let log = session.log_text();
    assert!(log.contains("tools/call echo"), "{log}");
    session.supervisor.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_registered_name_wins_over_the_mcp_tool() {
    let dir = std::env::temp_dir().join(format!(
        "agentsmith-mcp-collide-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("peer.log");
    let mut env = BTreeMap::new();
    env.insert("MCP_PEER_LOG".into(), log.display().to_string());
    let registry = ToolRegistry::new();
    registry
        .register(Arc::new(Fixed {
            name: "fs__echo".into(),
            body: "builtin-wins".into(),
        }))
        .unwrap();
    let mut servers = BTreeMap::new();
    servers.insert(
        "fs".into(),
        McpServerConfig {
            command: peer_bin(),
            args: Vec::new(),
            connect_timeout_secs: 5,
            call_timeout_secs: 5,
            env,
            ..McpServerConfig::default()
        },
    );
    let supervisor = McpSupervisor::new(McpSupervisorConfig {
        registry: registry.clone(),
        servers,
        max_output_bytes: 32 * 1024,
        shutdown_grace: Duration::from_secs(15),
        cancel: CancellationToken::new(),
    });
    supervisor.poll().await.unwrap();
    assert_eq!(supervisor.tools_skipped("fs"), 4);
    assert_eq!(
        registry.get("fs__echo").unwrap().definition().description,
        "builtin"
    );
    let ctx = ToolCtx {
        run_id: agentsmith_core::id::new_id(),
        workdir: dir.clone(),
        cancel: CancellationToken::new(),
    };
    let out = registry
        .get("fs__echo")
        .unwrap()
        .call(ctx, json!({ "text": "hi" }))
        .await
        .unwrap();
    assert_eq!(out.content, "builtin-wins");
    let read = registry.get("fs__read_file").unwrap();
    let read = read
        .call(
            ToolCtx {
                run_id: agentsmith_core::id::new_id(),
                workdir: dir.clone(),
                cancel: CancellationToken::new(),
            },
            json!({}),
        )
        .await
        .unwrap();
    assert_eq!(read.content, "mcp-read");
    let text = std::fs::read_to_string(&log).unwrap_or_default();
    assert!(
        !text.lines().any(|line| line == "tools/call echo"),
        "{text}"
    );
    assert!(text.contains("tools/call read_file"), "{text}");
    supervisor.shutdown().await.unwrap();
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dead_child_unregisters_and_later_calls_are_mcp_down() {
    let session = Session::start("ok", false, 2).await;
    let pid = session.supervisor.pid("fs").await.expect("recorded pid");
    let status = std::process::Command::new("kill")
        .args(["-KILL", &pid.to_string()])
        .status()
        .unwrap();
    assert!(status.success());
    // Do not poll here. poll would reconnect a working command.
    let (kind, message) = tool_err(
        session
            .supervisor
            .call(
                "fs",
                "echo",
                json!({ "text": "hi" }),
                CancellationToken::new(),
            )
            .await
            .unwrap_err(),
    );
    assert_eq!(kind, ToolErrorKind::McpDown);
    assert_eq!(message, "mcp server down");
    let (kind, message) = tool_err(
        session
            .supervisor
            .call("fs", "read_file", json!({}), CancellationToken::new())
            .await
            .unwrap_err(),
    );
    assert_eq!(kind, ToolErrorKind::McpDown);
    assert_eq!(message, "mcp server down");
    assert!(session
        .registry
        .names()
        .iter()
        .all(|name| !name.starts_with("fs__")));
    session.supervisor.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn input_required_is_one_call_and_does_not_follow_up() {
    let session = Session::start("input-required", false, 5).await;
    let (kind, message) = tool_err(
        session
            .call("echo", json!({ "text": "hi" }))
            .await
            .unwrap_err(),
    );
    assert_eq!(kind, ToolErrorKind::Execution);
    assert_eq!(message, "mcp follow-up not supported");
    let log = session.log_text();
    assert_eq!(log.matches("tools/call ").count(), 1, "{log}");
    session.supervisor.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_changed_replaces_the_server_tools() {
    let session = Session::start("ok", true, 5).await;
    assert!(session.registry.get("fs__echo").is_some());
    let mut replaced = false;
    for _ in 0..40 {
        session.supervisor.poll().await.unwrap();
        if session.registry.get("fs__echo2").is_some() && session.registry.get("fs__echo").is_none()
        {
            replaced = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(replaced, "tools: {:?}", session.registry.names());
    let out = session
        .call("echo2", json!({ "text": "next" }))
        .await
        .unwrap();
    assert_eq!(out.content, "next");
    session.supervisor.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_closes_stdin_and_does_not_signal_at_two_seconds() {
    let session = Session::start("ok", false, 5).await;
    assert!(session.supervisor.pid("fs").await.is_some());
    let started = Instant::now();
    session.supervisor.shutdown().await.unwrap();
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(2),
        "close_with_timeout took {elapsed:?}"
    );
    let log = session.log_text();
    assert!(log.contains("stdin-close"), "{log}");
    assert!(!log.contains("SIGTERM"), "{log}");
}
