//! Test fixture. Newline-delimited JSON-RPC on stdio, not a product entry point.
//! Logs go to `MCP_PEER_LOG` so they never share stdout with the protocol.

use std::io::{self, BufRead, Write};
use std::os::unix::io::AsRawFd;
use std::process::ExitCode;
use std::sync::atomic::{AtomicI32, Ordering};

static LOG_FD: AtomicI32 = AtomicI32::new(-1);

extern "C" fn on_sigterm(_: libc::c_int) {
    let fd = LOG_FD.load(Ordering::Relaxed);
    if fd < 0 {
        return;
    }
    let msg = b"SIGTERM\n";
    unsafe {
        libc::write(fd, msg.as_ptr().cast(), msg.len());
    }
}

fn main() -> ExitCode {
    // A closed stdout must not kill the process before stdin EOF is recorded.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
        libc::signal(libc::SIGTERM, on_sigterm as *const () as libc::sighandler_t);
    }
    let mut log = match std::env::var_os("MCP_PEER_LOG") {
        Some(path) => std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok(),
        None => None,
    };
    if let Some(file) = log.as_ref() {
        // Dup so the handler's fd stays open even if this File is closed.
        let fd = unsafe { libc::dup(file.as_raw_fd()) };
        if fd >= 0 {
            LOG_FD.store(fd, Ordering::Relaxed);
        }
    }
    write_log(&mut log, &format!("pid {}", std::process::id()));

    let mode = std::env::args().nth(1).unwrap_or_else(|| "ok".to_string());
    let notify = std::env::var("MCP_PEER_NOTIFY").ok().as_deref() == Some("1");
    let mut lists = 0u32;
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut lines = stdin.lock();
    let mut line = String::new();
    loop {
        line.clear();
        match lines.read_line(&mut line) {
            Ok(0) => {
                write_log(&mut log, "stdin-close");
                return ExitCode::SUCCESS;
            }
            Ok(_) => {}
            Err(_) => {
                write_log(&mut log, "stdin-close");
                return ExitCode::SUCCESS;
            }
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            continue;
        };
        let method = msg.get("method").and_then(|value| value.as_str());
        let id = msg.get("id").filter(|value| !value.is_null());
        let Some(method) = method else {
            continue;
        };
        let Some(id) = id else {
            continue;
        };
        match method {
            "initialize" if mode == "fail-init" => {
                write_error(&mut stdout, id, -32603, "initialize failed");
            }
            "initialize" => {
                write_result(
                    &mut stdout,
                    id,
                    serde_json::json!({
                        "protocolVersion": "2025-11-25",
                        "capabilities": { "tools": { "listChanged": true } },
                        "serverInfo": { "name": "peer", "version": "0.0.1" }
                    }),
                );
            }
            "ping" => write_result(&mut stdout, id, serde_json::json!({})),
            "tools/list" => {
                let tools = if lists == 0 {
                    tool_set(false)
                } else {
                    tool_set(true)
                };
                lists = lists.saturating_add(1);
                write_result(&mut stdout, id, serde_json::json!({ "tools": tools }));
                if lists == 1 && notify {
                    let _ = writeln!(
                        stdout,
                        "{}",
                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "method": "notifications/tools/list_changed"
                        })
                    );
                    let _ = stdout.flush();
                }
            }
            "tools/call" => {
                let name = msg
                    .get("params")
                    .and_then(|params| params.get("name"))
                    .and_then(|name| name.as_str())
                    .unwrap_or("");
                write_log(&mut log, &format!("tools/call {name}"));
                let arguments = msg
                    .get("params")
                    .and_then(|params| params.get("arguments"))
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                match mode.as_str() {
                    "input-required" => write_result(
                        &mut stdout,
                        id,
                        serde_json::json!({
                            "resultType": "input_required",
                            "requestState": "stay"
                        }),
                    ),
                    "task" => write_result(
                        &mut stdout,
                        id,
                        serde_json::json!({
                            "resultType": "task",
                            "taskId": "t1",
                            "status": "working",
                            "createdAt": "2026-01-01T00:00:00Z",
                            "lastUpdatedAt": "2026-01-01T00:00:00Z",
                            "ttlMs": null
                        }),
                    ),
                    _ => write_result(&mut stdout, id, call_result(name, &arguments)),
                }
            }
            _ => write_error(&mut stdout, id, -32601, "method not found"),
        }
    }
}

fn tool_set(replaced: bool) -> Vec<serde_json::Value> {
    let schema = serde_json::json!({ "type": "object", "additionalProperties": true });
    let long = "a".repeat(49);
    let echo = if replaced { "echo2" } else { "echo" };
    [echo, "read_file", "boom", "", long.as_str(), "has space"]
        .into_iter()
        .map(|name| {
            serde_json::json!({
                "name": name,
                "description": name,
                "inputSchema": schema.clone()
            })
        })
        .collect()
}

fn call_result(name: &str, arguments: &serde_json::Value) -> serde_json::Value {
    match name {
        "read_file" => serde_json::json!({
            "content": [{ "type": "text", "text": "mcp-read" }],
            "isError": false
        }),
        "boom" => serde_json::json!({
            "content": [{ "type": "text", "text": "nope" }],
            "isError": true
        }),
        _ => {
            let text = arguments
                .get("text")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let mut content = vec![serde_json::json!({ "type": "text", "text": text })];
            if arguments.get("image").and_then(|value| value.as_bool()) == Some(true) {
                content.push(serde_json::json!({
                    "type": "image",
                    "data": "a",
                    "mimeType": "image/png"
                }));
            }
            if arguments.get("resource").and_then(|value| value.as_bool()) == Some(true) {
                content.push(serde_json::json!({
                    "type": "resource",
                    "resource": { "uri": "file:///x", "text": "body" }
                }));
            }
            serde_json::json!({ "content": content, "isError": false })
        }
    }
}

fn write_result(stdout: &mut impl Write, id: &serde_json::Value, result: serde_json::Value) {
    let _ = writeln!(
        stdout,
        "{}",
        serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result })
    );
    let _ = stdout.flush();
}

fn write_error(stdout: &mut impl Write, id: &serde_json::Value, code: i32, message: &str) {
    let _ = writeln!(
        stdout,
        "{}",
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": code, "message": message }
        })
    );
    let _ = stdout.flush();
}

fn write_log(log: &mut Option<std::fs::File>, line: &str) {
    let Some(file) = log else {
        return;
    };
    let _ = writeln!(file, "{line}");
    let _ = file.flush();
}
