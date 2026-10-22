use std::process::ExitStatus;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};

use agentsmith_core::{Error, ToolErrorKind};

use super::{
    cap_output, clamp_output_bytes, DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_SHELL_TIMEOUT_SECS,
    MAX_SHELL_TIMEOUT_SECS,
};
use crate::tool::{ensure_not_cancelled, tool_error, Tool, ToolCtx, ToolDefinition, ToolOutput};

const NAME: &str = "shell";

/// SIGTERM, then SIGKILL if the group is still alive after this long.
const KILL_GRACE: Duration = Duration::from_secs(2);

/// Unsandboxed `sh -c` in the workdir. The file-tool jail does not apply.
#[derive(Debug)]
pub struct Shell {
    timeout: Duration,
    max_timeout: Duration,
    max_output_bytes: u64,
}

impl Default for Shell {
    fn default() -> Self {
        Self::new(
            Duration::from_secs(DEFAULT_SHELL_TIMEOUT_SECS),
            Duration::from_secs(MAX_SHELL_TIMEOUT_SECS),
            DEFAULT_MAX_OUTPUT_BYTES,
        )
    }
}

impl Shell {
    /// `timeout` is the default when the call omits `timeout_secs`.
    /// `max_timeout` caps a per-call override and is itself clamped to 120s.
    pub fn new(timeout: Duration, max_timeout: Duration, max_output_bytes: u64) -> Self {
        let ceiling = Duration::from_secs(MAX_SHELL_TIMEOUT_SECS);
        let max_timeout = if max_timeout > ceiling {
            ceiling
        } else {
            max_timeout
        };
        let timeout = if timeout > max_timeout {
            max_timeout
        } else {
            timeout
        };
        Self {
            timeout,
            max_timeout,
            max_output_bytes: clamp_output_bytes(max_output_bytes),
        }
    }

    fn timeout_for(&self, requested: Option<u64>) -> Result<Duration, Error> {
        let Some(secs) = requested else {
            return Ok(self.timeout);
        };
        if secs < 1 {
            return Err(tool_error(
                NAME,
                ToolErrorKind::InvalidArgs,
                "timeout_secs must be >= 1",
            ));
        }
        let requested = Duration::from_secs(secs);
        if requested > self.max_timeout {
            return Err(tool_error(
                NAME,
                ToolErrorKind::InvalidArgs,
                format!("timeout_secs exceeds {}", self.max_timeout.as_secs().max(1)),
            ));
        }
        Ok(requested)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    command: String,
    timeout_secs: Option<u64>,
}

enum WaitOutcome {
    Finished(ExitStatus),
    TimedOut,
    Cancelled,
    WaitFailed(std::io::Error),
}

/// Kills the child's process group if the call future is dropped mid-flight.
struct KillGroup {
    pid: u32,
    armed: bool,
}

impl Drop for KillGroup {
    fn drop(&mut self) {
        if self.armed {
            signal_group(self.pid, libc::SIGKILL);
        }
    }
}

#[async_trait::async_trait]
impl Tool for Shell {
    fn definition(&self) -> ToolDefinition {
        let max_secs = self.max_timeout.as_secs().clamp(1, MAX_SHELL_TIMEOUT_SECS);
        ToolDefinition {
            name: NAME.to_string(),
            description: "Run a shell command in the working directory. Non-zero exit is a tool error, not a fatal run error. The command is not sandboxed. Confinement is an external sandbox around the process, which this tool does not create.".to_string(),
            parameters: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["command"],
                "properties": {
                    "command": { "type": "string" },
                    "timeout_secs": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": max_secs
                    }
                }
            }),
        }
    }

    async fn call(&self, ctx: ToolCtx, arguments: Value) -> Result<ToolOutput, Error> {
        ensure_not_cancelled(NAME, &ctx)?;
        let args: Args = super::parse_args(NAME, arguments)?;
        let timeout = self.timeout_for(args.timeout_secs)?;
        ensure_not_cancelled(NAME, &ctx)?;

        let mut child = Command::new("sh")
            .arg("-c")
            .arg(&args.command)
            .current_dir(&ctx.workdir)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            // New process group whose id is the child's pid, so killpg reaches
            // grandchildren and not the harness.
            .process_group(0)
            .spawn()
            .map_err(|err| tool_error(NAME, ToolErrorKind::Execution, format!("spawn: {err}")))?;
        let pid = child
            .id()
            .ok_or_else(|| tool_error(NAME, ToolErrorKind::Execution, "shell child has no pid"))?;
        let mut guard = KillGroup { pid, armed: true };
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| tool_error(NAME, ToolErrorKind::Execution, "stdout was not piped"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| tool_error(NAME, ToolErrorKind::Execution, "stderr was not piped"))?;
        let cap = usize::try_from(self.max_output_bytes).unwrap_or(usize::MAX);
        let stdout_task = tokio::spawn(read_capped(stdout, cap));
        let stderr_task = tokio::spawn(read_capped(stderr, cap));

        let outcome = tokio::select! {
            biased;
            _ = ctx.cancel.cancelled() => WaitOutcome::Cancelled,
            waited = child.wait() => match waited {
                Ok(status) => WaitOutcome::Finished(status),
                Err(err) => WaitOutcome::WaitFailed(err),
            },
            _ = tokio::time::sleep(timeout) => WaitOutcome::TimedOut,
        };

        if !matches!(outcome, WaitOutcome::Finished(_)) {
            stop_group(&mut child, pid).await;
        }
        guard.armed = false;

        let stdout = collect(stdout_task).await;
        let stderr = collect(stderr_task).await;
        let combined = combine(&stdout, &stderr);
        let capped = cap_output(&combined, self.max_output_bytes, false);

        match outcome {
            WaitOutcome::Cancelled => Err(tool_error(
                NAME,
                ToolErrorKind::Cancelled,
                if capped.is_empty() {
                    "cancelled".to_string()
                } else {
                    capped
                },
            )),
            WaitOutcome::TimedOut => Err(tool_error(
                NAME,
                ToolErrorKind::Timeout,
                if capped.is_empty() {
                    "timed out".to_string()
                } else {
                    capped
                },
            )),
            WaitOutcome::WaitFailed(err) => Err(tool_error(
                NAME,
                ToolErrorKind::Execution,
                format!("wait: {err}"),
            )),
            WaitOutcome::Finished(status) if status.success() => Ok(ToolOutput { content: capped }),
            WaitOutcome::Finished(status) => {
                let message = if capped.is_empty() {
                    match status.code() {
                        Some(code) => format!("command exited with status {code}"),
                        None => "command exited with a signal".to_string(),
                    }
                } else {
                    capped
                };
                Err(tool_error(NAME, ToolErrorKind::Execution, message))
            }
        }
    }
}

/// TERM the group, wait up to 2s, then KILL. The leader exiting is not enough:
/// a grandchild may ignore TERM and keep the group alive.
async fn stop_group(child: &mut Child, pid: u32) {
    signal_group(pid, libc::SIGTERM);
    let reaped = tokio::time::timeout(KILL_GRACE, child.wait()).await;
    let leader_reaped = matches!(reaped, Ok(Ok(_)));
    if !leader_reaped || group_alive(pid) {
        signal_group(pid, libc::SIGKILL);
        if !leader_reaped {
            let _ = tokio::time::timeout(KILL_GRACE, child.wait()).await;
        }
        let deadline = tokio::time::Instant::now() + Duration::from_millis(500);
        while group_alive(pid) && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

fn signal_group(pid: u32, sig: i32) {
    let Ok(pgid) = i32::try_from(pid) else {
        return;
    };
    if pgid <= 0 {
        return;
    }
    // SAFETY: `pgid` is the process group of a child spawned with
    // `process_group(0)`, so it is that child's pid and not 0 (which would
    // signal the harness). `sig` is SIGTERM, SIGKILL, or checked by the caller.
    let _ = unsafe { libc::killpg(pgid, sig) };
}

fn group_alive(pid: u32) -> bool {
    let Ok(pgid) = i32::try_from(pid) else {
        return false;
    };
    if pgid <= 0 {
        return false;
    }
    // SAFETY: signal 0 only probes the group; it does not deliver a signal.
    let rc = unsafe { libc::killpg(pgid, 0) };
    if rc == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

async fn read_capped(mut reader: impl AsyncRead + Unpin, cap: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        match reader.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if out.len() < cap {
                    let room = cap - out.len();
                    out.extend_from_slice(&buf[..n.min(room)]);
                }
            }
        }
    }
    out
}

async fn collect(mut task: tokio::task::JoinHandle<Vec<u8>>) -> Vec<u8> {
    match tokio::time::timeout(Duration::from_secs(1), &mut task).await {
        Ok(Ok(buf)) => buf,
        Ok(Err(_)) => Vec::new(),
        Err(_) => {
            task.abort();
            Vec::new()
        }
    }
}

fn combine(stdout: &[u8], stderr: &[u8]) -> String {
    let mut out = String::from_utf8_lossy(stdout).into_owned();
    out.push_str(&String::from_utf8_lossy(stderr));
    out
}
