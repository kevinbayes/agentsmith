use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use agentsmith_core::id::new_id;
use agentsmith_tools::{
    register_builtins, BuiltinLimits, Error, Finish, ListDir, ReadFile, Shell, Tool, ToolCtx,
    ToolDefinition, ToolErrorKind, ToolOutput, ToolRegistry, WriteFile, DEFAULT_MAX_OUTPUT_BYTES,
    MAX_DIR_ENTRIES, MAX_OUTPUT_BYTES_CEILING, MAX_READ_BYTES_CEILING, MAX_WRITE_BYTES_CEILING,
};

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "agentsmith-tools-{}-{nanos}-{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn ctx(dir: &Path) -> ToolCtx {
    ctx_cancel(dir, CancellationToken::new())
}

fn ctx_cancel(dir: &Path, cancel: CancellationToken) -> ToolCtx {
    ToolCtx {
        run_id: new_id(),
        workdir: dir.to_path_buf(),
        cancel,
    }
}

fn kind_of(err: &Error) -> (ToolErrorKind, &str) {
    match err {
        Error::Tool { kind, message, .. } => (*kind, message),
        other => panic!("expected tool error, got {other}"),
    }
}

fn read_tool(max_read: u64, max_output: u64) -> ReadFile {
    ReadFile::new(max_read, max_output)
}

#[tokio::test]
async fn file_tools_round_trip_inside_the_workdir() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    std::fs::create_dir(&work).unwrap();
    std::fs::create_dir(work.join("ok")).unwrap();
    let write = WriteFile::new(1024, DEFAULT_MAX_OUTPUT_BYTES);
    let read = read_tool(1024, DEFAULT_MAX_OUTPUT_BYTES);
    let list = ListDir::new(DEFAULT_MAX_OUTPUT_BYTES);

    let wrote = write
        .call(
            ctx(&work),
            json!({"path": "./note.txt", "contents": "hello"}),
        )
        .await
        .unwrap();
    assert_eq!(wrote.content, "wrote 5 bytes");
    assert_eq!(
        std::fs::read_to_string(work.join("note.txt")).unwrap(),
        "hello"
    );

    let got = read
        .call(ctx(&work), json!({"path": "ok/../note.txt"}))
        .await
        .unwrap();
    assert_eq!(got.content, "hello");

    let abs = work.join("note.txt");
    let got = read
        .call(ctx(&work), json!({"path": abs.to_str().unwrap()}))
        .await
        .unwrap();
    assert_eq!(got.content, "hello");

    std::os::unix::fs::symlink("ok", work.join("dirlink")).unwrap();
    let listing = list.call(ctx(&work), json!({"path": "."})).await.unwrap();
    let lines: Vec<&str> = listing.content.split('\n').collect();
    assert!(lines.contains(&"note.txt"));
    assert!(lines.contains(&"ok/"));
    // A symlink to a directory is not a directory entry. Do not follow it.
    assert!(lines.contains(&"dirlink"));
    assert!(!lines.contains(&"dirlink/"));
}

#[tokio::test]
async fn path_jail_blocks_dotdot_and_symlink_escape() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    let outside = dir.path().join("outside");
    std::fs::create_dir(&work).unwrap();
    std::fs::create_dir(&outside).unwrap();
    let secret = "SECRET_MARKER_do_not_leak";
    std::fs::write(outside.join("secret.txt"), secret).unwrap();
    std::os::unix::fs::symlink(outside.join("secret.txt"), work.join("link")).unwrap();
    std::os::unix::fs::symlink(&outside, work.join("out_dir")).unwrap();
    std::fs::write(work.join("inside.txt"), "inside").unwrap();
    std::os::unix::fs::symlink(work.join("inside.txt"), work.join("safe_link")).unwrap();
    std::os::unix::fs::symlink("inside.txt", work.join("rel_link")).unwrap();

    let read = read_tool(4096, 4096);
    let list = ListDir::new(4096);

    let escaped = read
        .call(ctx(&work), json!({"path": "../outside/secret.txt"}))
        .await
        .unwrap_err();
    let (kind, message) = kind_of(&escaped);
    assert_eq!(kind, ToolErrorKind::Execution);
    assert!(!message.contains(secret));

    let link = read
        .call(ctx(&work), json!({"path": "link"}))
        .await
        .unwrap_err();
    let (kind, message) = kind_of(&link);
    assert_eq!(kind, ToolErrorKind::Execution);
    assert!(!message.contains(secret));

    let listed = list
        .call(ctx(&work), json!({"path": "out_dir"}))
        .await
        .unwrap_err();
    let (kind, message) = kind_of(&listed);
    assert_eq!(kind, ToolErrorKind::Execution);
    assert!(!message.contains("secret.txt"));

    let through_parent = read
        .call(ctx(&work), json!({"path": "out_dir/secret.txt"}))
        .await
        .unwrap_err();
    let (kind, message) = kind_of(&through_parent);
    assert_eq!(kind, ToolErrorKind::Execution);
    assert_eq!(message, "path escapes the working directory");
    assert!(!message.contains(secret));

    let absolute = read
        .call(
            ctx(&work),
            json!({"path": outside.join("secret.txt").to_str().unwrap()}),
        )
        .await
        .unwrap_err();
    assert_eq!(kind_of(&absolute).0, ToolErrorKind::Execution);
    assert!(!kind_of(&absolute).1.contains(secret));

    let safe = read
        .call(ctx(&work), json!({"path": "safe_link"}))
        .await
        .unwrap();
    assert_eq!(safe.content, "inside");

    let relative = read
        .call(ctx(&work), json!({"path": "rel_link"}))
        .await
        .unwrap();
    assert_eq!(relative.content, "inside");
}

#[tokio::test]
async fn write_does_not_follow_an_escaping_or_broken_symlink() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    let outside = dir.path().join("outside");
    std::fs::create_dir(&work).unwrap();
    std::fs::create_dir(&outside).unwrap();
    std::fs::write(outside.join("secret.txt"), "original").unwrap();
    std::os::unix::fs::symlink(outside.join("secret.txt"), work.join("link")).unwrap();
    std::os::unix::fs::symlink(outside.join("missing.txt"), work.join("broken")).unwrap();
    std::os::unix::fs::symlink(&outside, work.join("out_dir")).unwrap();

    let write = WriteFile::new(1024, 1024);
    for path in ["link", "broken", "out_dir/pwned.txt"] {
        let err = write
            .call(ctx(&work), json!({"path": path, "contents": "pwned"}))
            .await
            .unwrap_err();
        assert_eq!(kind_of(&err).0, ToolErrorKind::Execution, "{path}");
        assert!(!kind_of(&err).1.contains("pwned"));
    }
    assert_eq!(
        std::fs::read_to_string(outside.join("secret.txt")).unwrap(),
        "original"
    );
    assert!(!outside.join("missing.txt").exists());
    assert!(!outside.join("pwned.txt").exists());
}

#[tokio::test]
async fn write_limit_does_not_create_a_partial_file() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    std::fs::create_dir(&work).unwrap();
    let write = WriteFile::new(4, 1024);
    let err = write
        .call(ctx(&work), json!({"path": "big.txt", "contents": "hello"}))
        .await
        .unwrap_err();
    assert_eq!(kind_of(&err).0, ToolErrorKind::Limit);
    assert!(!work.join("big.txt").exists());

    write
        .call(ctx(&work), json!({"path": "big.txt", "contents": "hey"}))
        .await
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(work.join("big.txt")).unwrap(),
        "hey"
    );
}

#[tokio::test]
async fn write_hard_ceiling_is_one_mibibyte() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    std::fs::create_dir(&work).unwrap();
    let write = WriteFile::new(u64::MAX, u64::MAX);
    let ok = "a".repeat(MAX_WRITE_BYTES_CEILING as usize);
    write
        .call(ctx(&work), json!({"path": "ceil.txt", "contents": ok}))
        .await
        .unwrap();
    assert_eq!(
        std::fs::metadata(work.join("ceil.txt")).unwrap().len(),
        MAX_WRITE_BYTES_CEILING
    );
    let too_big = "a".repeat(MAX_WRITE_BYTES_CEILING as usize + 1);
    let err = write
        .call(ctx(&work), json!({"path": "over.txt", "contents": too_big}))
        .await
        .unwrap_err();
    assert_eq!(kind_of(&err).0, ToolErrorKind::Limit);
    assert!(!work.join("over.txt").exists());
}

#[tokio::test]
async fn read_respects_constructor_cap_and_char_boundary() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    std::fs::create_dir(&work).unwrap();
    std::fs::write(work.join("exact.txt"), "abcd").unwrap();
    std::fs::write(work.join("over.txt"), "abcde").unwrap();
    // é is two bytes. A 3-byte cap must not split the second character.
    let chars = "éé";
    assert_eq!(chars.len(), 4);
    std::fs::write(work.join("utf.txt"), chars).unwrap();
    std::fs::write(work.join("bin.txt"), [0xff, 0xfe]).unwrap();

    let read = read_tool(4, 1024);
    assert_eq!(
        read.call(ctx(&work), json!({"path": "exact.txt"}))
            .await
            .unwrap()
            .content,
        "abcd"
    );
    assert_eq!(
        read.call(ctx(&work), json!({"path": "over.txt"}))
            .await
            .unwrap()
            .content,
        "abcd\n[truncated]"
    );

    let short = read_tool(3, 1024);
    assert_eq!(
        short
            .call(ctx(&work), json!({"path": "utf.txt"}))
            .await
            .unwrap()
            .content,
        "é\n[truncated]"
    );

    let err = read
        .call(ctx(&work), json!({"path": "bin.txt"}))
        .await
        .unwrap_err();
    assert_eq!(kind_of(&err).0, ToolErrorKind::Execution);
    assert!(!kind_of(&err).1.contains('\u{fffd}'));
}

#[tokio::test]
async fn read_and_list_do_not_block_on_a_fifo() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    std::fs::create_dir(&work).unwrap();
    let fifo = work.join("pipe");
    let status = std::process::Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .expect("mkfifo");
    assert!(status.success());

    let read = read_tool(1024, 1024);
    let started = Instant::now();
    let err = tokio::time::timeout(
        Duration::from_secs(3),
        read.call(ctx(&work), json!({"path": "pipe"})),
    )
    .await
    .expect("read_file blocked on a fifo")
    .unwrap_err();
    assert_eq!(kind_of(&err).0, ToolErrorKind::Execution);
    assert!(started.elapsed() < Duration::from_secs(3));

    let list = ListDir::new(1024);
    let err = tokio::time::timeout(
        Duration::from_secs(3),
        list.call(ctx(&work), json!({"path": "pipe"})),
    )
    .await
    .expect("list_dir blocked on a fifo")
    .unwrap_err();
    assert_eq!(kind_of(&err).0, ToolErrorKind::Execution);
}

#[tokio::test]
async fn read_output_cap_truncates_on_a_char_boundary() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    std::fs::create_dir(&work).unwrap();
    let text = "é".repeat(9);
    assert_eq!(text.len(), 18);
    std::fs::write(work.join("wide.txt"), &text).unwrap();
    let read = read_tool(1024, 16);
    let got = read
        .call(ctx(&work), json!({"path": "wide.txt"}))
        .await
        .unwrap();
    assert!(got.content.len() <= 16, "{}", got.content.len());
    assert!(got.content.ends_with("\n[truncated]"));
    assert!(got
        .content
        .trim_end_matches("\n[truncated]")
        .is_char_boundary(got.content.trim_end_matches("\n[truncated]").len()));
}

#[tokio::test]
async fn read_hard_ceiling_caps_output_at_32_kib() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    std::fs::create_dir(&work).unwrap();
    let big = vec![b'a'; (MAX_READ_BYTES_CEILING as usize) + 8];
    std::fs::write(work.join("big.txt"), &big).unwrap();
    let read = ReadFile::new(u64::MAX, u64::MAX);
    let got = read
        .call(ctx(&work), json!({"path": "big.txt"}))
        .await
        .unwrap();
    assert!(got.content.len() <= MAX_OUTPUT_BYTES_CEILING as usize);
    assert!(got.content.ends_with("[truncated]"));
}

#[tokio::test]
async fn invalid_json_arguments_are_invalid_args() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    std::fs::create_dir(&work).unwrap();
    let read = read_tool(64, 64);
    let cases = [
        Value::String("{".into()),
        Value::String("not json".into()),
        Value::String("[1]".into()),
        json!({}),
        json!({"path": 1}),
        json!({"path": "a", "extra": true}),
        json!(["a"]),
        json!(null),
    ];
    for arguments in cases {
        let err = read.call(ctx(&work), arguments.clone()).await.unwrap_err();
        let (kind, message) = kind_of(&err);
        assert_eq!(
            kind,
            ToolErrorKind::InvalidArgs,
            "arguments {arguments} produced {message}"
        );
    }
}

#[tokio::test]
async fn cancelled_token_does_not_start_work() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    let outside = dir.path().join("outside");
    std::fs::create_dir(&work).unwrap();
    std::fs::create_dir(&outside).unwrap();
    std::fs::write(outside.join("secret.txt"), "SECRET_MARKER").unwrap();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let read = read_tool(1024, 1024);
    let err = read
        .call(
            ctx_cancel(&work, cancel.clone()),
            json!({"path": "../outside/secret.txt"}),
        )
        .await
        .unwrap_err();
    let (kind, message) = kind_of(&err);
    assert_eq!(kind, ToolErrorKind::Cancelled);
    assert!(!message.contains("SECRET_MARKER"));

    let write = WriteFile::new(1024, 1024);
    let err = write
        .call(
            ctx_cancel(&work, cancel),
            json!({"path": "nope.txt", "contents": "x"}),
        )
        .await
        .unwrap_err();
    assert_eq!(kind_of(&err).0, ToolErrorKind::Cancelled);
    assert!(!work.join("nope.txt").exists());
}

#[tokio::test]
async fn list_dir_caps_entries_and_is_not_recursive() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    std::fs::create_dir_all(work.join("sub")).unwrap();
    std::fs::write(work.join("sub").join("hidden.txt"), "nope").unwrap();
    for i in 0..=MAX_DIR_ENTRIES {
        std::fs::write(work.join(format!("f{i:04}")), "x").unwrap();
    }
    let list = ListDir::new(MAX_OUTPUT_BYTES_CEILING);
    let got = list.call(ctx(&work), json!({"path": "."})).await.unwrap();
    assert!(got.content.ends_with("\n[truncated]"));
    assert!(!got.content.contains("hidden.txt"));
    let body = got.content.trim_end_matches("\n[truncated]");
    assert_eq!(body.lines().count(), MAX_DIR_ENTRIES);
}

#[tokio::test]
async fn duplicate_register_is_rejected_and_definitions_are_ordered() {
    let registry = ToolRegistry::new();
    registry.register(Arc::new(WriteFile::default())).unwrap();
    registry.register(Arc::new(Finish::default())).unwrap();
    registry.register(Arc::new(ReadFile::default())).unwrap();
    let err = registry.register(Arc::new(Finish::default())).unwrap_err();
    assert_eq!(kind_of(&err).0, ToolErrorKind::InvalidArgs);
    assert_eq!(
        registry.names(),
        vec![
            "finish".to_string(),
            "read_file".to_string(),
            "write_file".to_string()
        ]
    );
    let defined: Vec<_> = registry
        .definitions()
        .into_iter()
        .map(|def| def.name)
        .collect();
    assert_eq!(defined, registry.names());
}

#[tokio::test]
async fn unregister_prefix_removes_only_the_namespace() {
    let registry = ToolRegistry::new();
    registry.register(Arc::new(Named("read_file"))).unwrap();
    registry.register(Arc::new(Named("fs"))).unwrap();
    registry.register(Arc::new(Named("fs__read_file"))).unwrap();
    registry.register(Arc::new(Named("fast__other"))).unwrap();
    registry.unregister_prefix("fs");
    assert_eq!(
        registry.names(),
        vec![
            "fast__other".to_string(),
            "fs".to_string(),
            "read_file".to_string()
        ]
    );
}

struct Named(&'static str);

#[async_trait]
impl Tool for Named {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.0.to_string(),
            description: "test".to_string(),
            parameters: json!({"type": "object", "additionalProperties": false}),
        }
    }

    async fn call(&self, _ctx: ToolCtx, _arguments: Value) -> Result<ToolOutput, Error> {
        Ok(ToolOutput {
            content: self.0.to_string(),
        })
    }
}

struct Reenter {
    registry: ToolRegistry,
}

#[async_trait]
impl Tool for Reenter {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "reenter".to_string(),
            description: "test".to_string(),
            parameters: json!({"type": "object", "additionalProperties": false}),
        }
    }

    async fn call(&self, _ctx: ToolCtx, _arguments: Value) -> Result<ToolOutput, Error> {
        self.registry.register(Arc::new(Finish::new(64)))?;
        assert!(self.registry.get("reenter").is_some());
        let _ = self.registry.definitions();
        tokio::task::yield_now().await;
        self.registry.unregister_prefix("nope");
        Ok(ToolOutput {
            content: "ok".to_string(),
        })
    }
}

#[tokio::test]
async fn tool_can_reenter_the_registry_during_call() {
    let registry = ToolRegistry::new();
    registry
        .register(Arc::new(Reenter {
            registry: registry.clone(),
        }))
        .unwrap();
    let tool = registry.get("reenter").unwrap();
    let output = tokio::time::timeout(
        Duration::from_secs(2),
        tool.call(ctx(&std::env::temp_dir()), json!({})),
    )
    .await
    .expect("deadlock: registry lock held across call")
    .unwrap();
    assert_eq!(output.content, "ok");
    assert_eq!(
        registry.names(),
        vec!["finish".to_string(), "reenter".to_string()]
    );
}

#[tokio::test]
async fn finish_returns_the_answer_without_touching_the_filesystem() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    std::fs::create_dir(&work).unwrap();
    let before = std::fs::read_dir(&work).unwrap().count();
    let finish = Finish::new(1024);
    let got = finish
        .call(ctx(&work), json!({"answer": "all done"}))
        .await
        .unwrap();
    assert_eq!(got.content, "all done");
    assert_eq!(std::fs::read_dir(&work).unwrap().count(), before);

    let tiny = Finish::new(16);
    let got = tiny
        .call(ctx(&work), json!({"answer": "abcdefghijklmnopqrstuvwxyz"}))
        .await
        .unwrap();
    assert_eq!(got.content, "abcd\n[truncated]");
    assert!(got.content.len() <= 16);
}

#[tokio::test]
async fn shell_echo_returns_stdout_and_nonzero_is_execution() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    std::fs::create_dir(&work).unwrap();
    let shell = Shell::new(Duration::from_secs(5), Duration::from_secs(5), 1024);
    let got = shell
        .call(ctx(&work), json!({"command": "echo hello"}))
        .await
        .unwrap();
    assert_eq!(got.content, "hello\n");

    let err = shell
        .call(ctx(&work), json!({"command": "echo no; exit 1"}))
        .await
        .unwrap_err();
    let (kind, message) = kind_of(&err);
    assert_eq!(kind, ToolErrorKind::Execution);
    assert!(message.contains("no"), "{message}");

    let err = shell
        .call(ctx(&work), json!({"command": "exit 1"}))
        .await
        .unwrap_err();
    let (kind, message) = kind_of(&err);
    assert_eq!(kind, ToolErrorKind::Execution);
    assert!(message.contains("status 1"), "{message}");
}

#[tokio::test]
async fn shell_timeout_kills_the_process_group() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    std::fs::create_dir(&work).unwrap();
    let pid_path = work.join("sleep.pid");
    let command = format!(
        "trap '' TERM; sleep 30 & echo $! > {}; wait",
        sh_quote(pid_path.to_str().unwrap())
    );
    // Sub-second default so the override is not required. The child ignores
    // TERM, so the call must spend the 2s SIGKILL grace and still return.
    let shell = Shell::new(Duration::from_millis(400), Duration::from_secs(5), 1024);
    let started = Instant::now();
    let result = tokio::time::timeout(
        Duration::from_secs(12),
        shell.call(ctx(&work), json!({"command": command})),
    )
    .await;
    let elapsed = started.elapsed();
    let err = match result {
        Ok(inner) => inner.expect_err("timeout must be a tool error"),
        Err(_) => {
            kill_path(&pid_path);
            panic!("shell call hung; elapsed {elapsed:?}");
        }
    };
    let (kind, message) = kind_of(&err);
    assert_eq!(kind, ToolErrorKind::Timeout, "{message}");
    assert!(
        elapsed >= Duration::from_millis(2_000),
        "expected the SIGKILL grace, elapsed {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_secs(12),
        "kill took too long: {elapsed:?}"
    );

    let pid_text = std::fs::read_to_string(&pid_path).expect("child pid was not recorded");
    let pid: u32 = pid_text.trim().parse().expect("pid");
    let mut dead = false;
    for _ in 0..50 {
        if !pid_alive(pid) {
            dead = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    if !dead {
        kill_pid(pid);
    }
    assert!(dead, "process group member {pid} still alive");
}

#[tokio::test]
async fn shell_background_sleep_is_killed_when_the_leader_exits() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    std::fs::create_dir(&work).unwrap();
    let pid_path = work.join("sleep.pid");
    // `sh -c 'sleep 30 &'`: the leader exits immediately and the sleep stays
    // in its process group. Record $! so the test can see that grandchild.
    let command = format!(
        "sleep 30 & echo $! > {}",
        sh_quote(pid_path.to_str().unwrap())
    );
    let shell = Shell::new(Duration::from_secs(5), Duration::from_secs(5), 1024);
    let started = Instant::now();
    let result = tokio::time::timeout(
        Duration::from_secs(10),
        shell.call(ctx(&work), json!({"command": command})),
    )
    .await;
    let elapsed = started.elapsed();
    let output = match result {
        Ok(inner) => inner.expect("background sleep must not fail the command"),
        Err(_) => {
            kill_path(&pid_path);
            panic!("shell call hung; elapsed {elapsed:?}");
        }
    };
    assert!(
        elapsed < Duration::from_secs(8),
        "background sleep was left running: {elapsed:?} output {:?}",
        output.content
    );
    let pid_text = std::fs::read_to_string(&pid_path).expect("child pid was not recorded");
    let pid: u32 = pid_text.trim().parse().expect("pid");
    let mut dead = false;
    for _ in 0..50 {
        if !pid_alive(pid) {
            dead = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    if !dead {
        kill_pid(pid);
    }
    assert!(dead, "background sleep {pid} still alive");
}

#[tokio::test]
async fn shell_cancel_does_not_wait_out_the_command() {
    let dir = TempDir::new();
    let work = dir.path().join("work");
    std::fs::create_dir(&work).unwrap();
    let cancel = CancellationToken::new();
    let flag = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(150)).await;
        flag.cancel();
    });
    let shell = Shell::default();
    let started = Instant::now();
    let err = shell
        .call(ctx_cancel(&work, cancel), json!({"command": "sleep 30"}))
        .await
        .unwrap_err();
    assert_eq!(kind_of(&err).0, ToolErrorKind::Cancelled);
    assert!(started.elapsed() < Duration::from_secs(10));
}

#[tokio::test]
async fn register_builtins_orders_names_and_skips_shell_unless_enabled() {
    let registry = ToolRegistry::new();
    register_builtins(&registry, &BuiltinLimits::default()).unwrap();
    assert_eq!(
        registry.names(),
        vec![
            "finish".to_string(),
            "list_dir".to_string(),
            "read_file".to_string(),
            "write_file".to_string()
        ]
    );

    let limits = BuiltinLimits {
        enable_shell: true,
        ..BuiltinLimits::default()
    };
    let registry = ToolRegistry::new();
    register_builtins(&registry, &limits).unwrap();
    assert_eq!(
        registry.names(),
        vec![
            "finish".to_string(),
            "list_dir".to_string(),
            "read_file".to_string(),
            "shell".to_string(),
            "write_file".to_string()
        ]
    );
    let shell = registry.get("shell").unwrap().definition();
    assert_eq!(shell.parameters["additionalProperties"], json!(false));
    assert_eq!(
        shell.parameters["properties"]["timeout_secs"]["maximum"],
        json!(120)
    );
    assert_eq!(shell.parameters["required"], json!(["command"]));

    let read = registry.get("read_file").unwrap().definition();
    assert_eq!(read.parameters["additionalProperties"], json!(false));
    assert_eq!(read.parameters["required"], json!(["path"]));
}

fn sh_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn pid_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn kill_pid(pid: u32) {
    let _ = std::process::Command::new("kill")
        .args(["-KILL", &format!("-{pid}")])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    let _ = std::process::Command::new("kill")
        .args(["-KILL", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

fn kill_path(path: &Path) {
    if let Ok(text) = std::fs::read_to_string(path) {
        if let Ok(pid) = text.trim().parse::<u32>() {
            kill_pid(pid);
        }
    }
}
