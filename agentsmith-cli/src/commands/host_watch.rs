//! Host process sample for an AHSP report. Not part of a run snapshot.

use std::path::{Path, PathBuf};
use std::time::Duration;

use agentsmith_core::config::HostWatch;
use agentsmith_core::redact::{cut_display, redact, take_chars};
use agentsmith_protocol::{ProcessWatch, RunningInstance};

const VERSION_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_INSTANCES: usize = 16;

/// `(processes, truncated)`. Empty `watch` is an empty list and not truncated.
pub(crate) async fn sample(watch: &[HostWatch]) -> (Vec<ProcessWatch>, bool) {
    if watch.is_empty() {
        return (Vec::new(), false);
    }
    let located = locate_all(watch).await;
    let procs = list_processes().await;
    let mut truncated = false;
    let mut processes = Vec::with_capacity(watch.len());
    for (entry, located) in watch.iter().zip(located) {
        let base = basename(&entry.binary);
        let mut instances = Vec::new();
        if located.installed {
            for proc in &procs {
                if !process_matches(proc, &base, entry.match_node) {
                    continue;
                }
                if instances.len() == MAX_INSTANCES {
                    truncated = true;
                    break;
                }
                instances.push(RunningInstance {
                    pid: proc.pid,
                    name: take_chars(&proc.name, 64).to_string(),
                    cmd_summary: cut_display(&redact(&proc.cmd)),
                });
            }
        }
        processes.push(ProcessWatch {
            name: take_chars(&entry.name, 64).to_string(),
            binary: take_chars(&entry.binary, 256).to_string(),
            installed: located.installed,
            version: located.version,
            path: located
                .path
                .map(|path| take_chars(&path.to_string_lossy(), 512).to_string()),
            running_instances: instances,
        });
    }
    (processes, truncated)
}

struct Located {
    installed: bool,
    path: Option<PathBuf>,
    version: Option<String>,
}

async fn locate_all(watch: &[HostWatch]) -> Vec<Located> {
    let mut set = tokio::task::JoinSet::new();
    for (index, entry) in watch.iter().enumerate() {
        let binary = entry.binary.clone();
        set.spawn(async move { (index, locate(&binary).await) });
    }
    let mut located = Vec::new();
    located.resize_with(watch.len(), || Located {
        installed: false,
        path: None,
        version: None,
    });
    while let Some(joined) = set.join_next().await {
        if let Ok((index, item)) = joined {
            located[index] = item;
        }
    }
    located
}

async fn locate(binary: &str) -> Located {
    let Some(path) = which(binary) else {
        return Located {
            installed: false,
            path: None,
            version: None,
        };
    };
    let version = version_of(&path).await;
    Located {
        installed: true,
        path: Some(path),
        version,
    }
}

fn which(binary: &str) -> Option<PathBuf> {
    let path = Path::new(binary);
    if binary.contains('/') || binary.contains('\\') {
        return is_exec(path).then(|| path.to_path_buf());
    }
    let paths = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&paths) {
        let candidate = dir.join(path);
        if is_exec(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn is_exec(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

async fn version_of(path: &Path) -> Option<String> {
    let mut cmd = tokio::process::Command::new(path);
    cmd.arg("--version")
        .env_clear()
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    if let Some(path_env) = std::env::var_os("PATH") {
        cmd.env("PATH", path_env);
    }
    let finished = tokio::time::timeout(VERSION_TIMEOUT, cmd.output()).await;
    let output = match finished {
        Ok(Ok(output)) if output.status.success() => output,
        _ => return None,
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let line = first_line(&stdout).or_else(|| first_line(&stderr))?;
    let cut = cut_display(&redact(&line));
    if cut.is_empty() {
        None
    } else {
        Some(cut)
    }
}

fn first_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(str::to_string)
}

fn basename(binary: &str) -> String {
    Path::new(binary)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| binary.to_string())
}

struct ProcInfo {
    pid: u32,
    name: String,
    argv0: String,
    cmd: String,
}

fn process_matches(proc: &ProcInfo, base: &str, match_node: bool) -> bool {
    if proc.name == base || proc.argv0 == base {
        return true;
    }
    // `comm` is 15 bytes. `node` plus the script name is the other shape.
    match_node
        && (proc.name == "node" || proc.argv0 == "node")
        && !base.is_empty()
        && proc.cmd.contains(base)
}

async fn list_processes() -> Vec<ProcInfo> {
    #[cfg(target_os = "linux")]
    {
        tokio::task::spawn_blocking(scan_proc)
            .await
            .unwrap_or_default()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Vec::new()
    }
}

#[cfg(target_os = "linux")]
fn scan_proc() -> Vec<ProcInfo> {
    // One pass over /proc. Revisit if a host has ~10k processes and this pass
    // exceeds the report interval.
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    let mut procs = Vec::new();
    for entry in entries.flatten() {
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
            continue;
        };
        let dir = entry.path();
        let comm = std::fs::read_to_string(dir.join("comm")).unwrap_or_default();
        let name = comm.trim().to_string();
        let raw = std::fs::read(dir.join("cmdline")).unwrap_or_default();
        let args: Vec<String> = raw
            .split(|byte| *byte == 0)
            .filter(|arg| !arg.is_empty())
            .map(|arg| String::from_utf8_lossy(arg).into_owned())
            .collect();
        let argv0 = args
            .first()
            .and_then(|arg| {
                Path::new(arg)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
            })
            .unwrap_or_default();
        procs.push(ProcInfo {
            pid,
            name,
            argv0,
            cmd: args.join(" "),
        });
    }
    procs.sort_by_key(|proc| proc.pid);
    procs
}
