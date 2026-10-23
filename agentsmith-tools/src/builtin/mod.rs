//! Builtins share one path jail and one output cap.
//!
//! `read_file`, `write_file`, and `list_dir` open from a workdir directory fd
//! with `openat2` (`RESOLVE_BENEATH`). The kernel follows symlinks that stay
//! inside that directory and rejects a component that resolves outside it.
//! Callers read, write, and list through the returned fd. `shell` does not
//! use this jail.

mod finish;
mod list_dir;
mod read_file;
mod shell;
mod write_file;

use std::collections::VecDeque;
use std::ffi::{CString, OsStr, OsString};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde_json::Value;

use agentsmith_core::{Error, ToolErrorKind};

use crate::registry::ToolRegistry;
use crate::tool::tool_error;

pub use finish::Finish;
pub use list_dir::ListDir;
pub use read_file::ReadFile;
pub use shell::Shell;
pub use write_file::WriteFile;

pub const DEFAULT_MAX_READ_BYTES: u64 = 64 * 1024;
pub const MAX_READ_BYTES_CEILING: u64 = 256 * 1024;
pub const DEFAULT_MAX_WRITE_BYTES: u64 = 256 * 1024;
pub const MAX_WRITE_BYTES_CEILING: u64 = 1024 * 1024;
pub const DEFAULT_MAX_OUTPUT_BYTES: u64 = 32 * 1024;
pub const MAX_OUTPUT_BYTES_CEILING: u64 = 32 * 1024;
pub const DEFAULT_SHELL_TIMEOUT_SECS: u64 = 30;
pub const MAX_SHELL_TIMEOUT_SECS: u64 = 120;
pub const MAX_DIR_ENTRIES: usize = 500;

const TRUNCATED_SUFFIX: &str = "\n[truncated]";

/// Limits passed to [`register_builtins`]. Constructors clamp to the ceilings.
#[derive(Clone, Debug)]
pub struct BuiltinLimits {
    pub max_read_bytes: u64,
    pub max_write_bytes: u64,
    pub max_output_bytes: u64,
    pub shell_timeout: Duration,
    pub shell_max_timeout: Duration,
    /// Shell is host execution. Callers turn it on; the default stays off.
    pub enable_shell: bool,
}

impl Default for BuiltinLimits {
    fn default() -> Self {
        Self {
            max_read_bytes: DEFAULT_MAX_READ_BYTES,
            max_write_bytes: DEFAULT_MAX_WRITE_BYTES,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            shell_timeout: Duration::from_secs(DEFAULT_SHELL_TIMEOUT_SECS),
            shell_max_timeout: Duration::from_secs(MAX_SHELL_TIMEOUT_SECS),
            enable_shell: false,
        }
    }
}

/// Register the always-on builtins, plus `shell` when `limits.enable_shell`.
pub fn register_builtins(registry: &ToolRegistry, limits: &BuiltinLimits) -> Result<(), Error> {
    registry.register(Arc::new(ReadFile::new(
        limits.max_read_bytes,
        limits.max_output_bytes,
    )))?;
    registry.register(Arc::new(WriteFile::new(
        limits.max_write_bytes,
        limits.max_output_bytes,
    )))?;
    registry.register(Arc::new(ListDir::new(limits.max_output_bytes)))?;
    registry.register(Arc::new(Finish::new(limits.max_output_bytes)))?;
    if limits.enable_shell {
        registry.register(Arc::new(Shell::new(
            limits.shell_timeout,
            limits.shell_max_timeout,
            limits.max_output_bytes,
        )))?;
    }
    Ok(())
}

pub(crate) fn clamp_output_bytes(value: u64) -> u64 {
    value.min(MAX_OUTPUT_BYTES_CEILING)
}

pub(crate) fn parse_args<T: DeserializeOwned>(tool: &str, arguments: Value) -> Result<T, Error> {
    let arguments = match arguments {
        Value::String(raw) => serde_json::from_str(&raw).map_err(|err| {
            tool_error(
                tool,
                ToolErrorKind::InvalidArgs,
                format!("invalid JSON arguments: {err}"),
            )
        })?,
        other => other,
    };
    // A one-field struct deserializes from a one-element array. Every builtin
    // schema is an object, so reject anything else before that happens.
    if !arguments.is_object() {
        return Err(tool_error(
            tool,
            ToolErrorKind::InvalidArgs,
            "arguments must be a JSON object",
        ));
    }
    serde_json::from_value(arguments).map_err(|err| {
        tool_error(
            tool,
            ToolErrorKind::InvalidArgs,
            format!("invalid arguments: {err}"),
        )
    })
}

/// Model-facing cap shared with builtin tools, so an MCP result cannot exceed
/// the same ceiling.
pub fn cap_tool_output(content: &str, max_bytes: u64) -> String {
    cap_output(content, max_bytes, false)
}

/// Cut `content` to the output cap. Truncation keeps a `\n[truncated]` suffix
/// when it fits, on a char boundary. `already_cut` forces the suffix even when
/// the remaining text is under the cap (the read cap, or a long directory).
pub(crate) fn cap_output(content: &str, max_bytes: u64, already_cut: bool) -> String {
    let max_bytes = usize::try_from(clamp_output_bytes(max_bytes)).unwrap_or(usize::MAX);
    if !already_cut && content.len() <= max_bytes {
        return content.to_string();
    }
    if already_cut && content.len().saturating_add(TRUNCATED_SUFFIX.len()) <= max_bytes {
        let mut out = String::with_capacity(content.len() + TRUNCATED_SUFFIX.len());
        out.push_str(content);
        out.push_str(TRUNCATED_SUFFIX);
        return out;
    }
    if max_bytes >= TRUNCATED_SUFFIX.len() {
        let end = floor_char_boundary(content, max_bytes - TRUNCATED_SUFFIX.len());
        let mut out = String::with_capacity(end + TRUNCATED_SUFFIX.len());
        out.push_str(&content[..end]);
        out.push_str(TRUNCATED_SUFFIX);
        return out;
    }
    let end = floor_char_boundary(content, max_bytes);
    content[..end].to_string()
}

fn floor_char_boundary(content: &str, max_bytes: usize) -> usize {
    let mut end = max_bytes.min(content.len());
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    end
}

pub(crate) async fn join_blocking<T>(
    tool: &str,
    handle: tokio::task::JoinHandle<Result<T, Error>>,
) -> Result<T, Error> {
    match handle.await {
        Ok(result) => result,
        Err(err) if err.is_panic() => {
            Err(tool_error(tool, ToolErrorKind::Execution, "task panicked"))
        }
        Err(err) => Err(tool_error(tool, ToolErrorKind::Execution, err.to_string())),
    }
}

/// Directory fd for the run workdir. User paths are opened relative to it.
pub(crate) struct WorkDirFd {
    fd: OwnedFd,
}

impl WorkDirFd {
    pub(crate) fn open(tool: &str, workdir: &Path) -> Result<Self, Error> {
        let c_path = CString::new(workdir.as_os_str().as_bytes()).map_err(|_| {
            tool_error(
                tool,
                ToolErrorKind::Execution,
                "workdir: path contains a null byte",
            )
        })?;
        let fd = unsafe {
            // SAFETY: `c_path` is a NUL-terminated workdir path the caller owns
            // for this call. O_DIRECTORY fails unless the inode is a directory.
            libc::open(
                c_path.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            let err = std::io::Error::last_os_error();
            return Err(tool_error(
                tool,
                ToolErrorKind::Execution,
                format!("workdir: {err}"),
            ));
        }
        Ok(Self {
            // SAFETY: `fd` is a fresh descriptor from `open`, or we already returned.
            fd: unsafe { OwnedFd::from_raw_fd(fd) },
        })
    }

    /// Path to pass to `openat2` on this fd. Absolute requests are prefix-stripped
    /// against the fd's path so `RESOLVE_BENEATH` can accept them; a path that
    /// does not sit under the fd is an escape. Relative requests are unchanged
    /// so a leading `..` is still rejected by the kernel.
    pub(crate) fn relative(&self, tool: &str, requested: &str) -> Result<PathBuf, Error> {
        if requested.is_empty() || requested.contains('\0') {
            return Err(tool_error(
                tool,
                ToolErrorKind::InvalidArgs,
                "path is empty",
            ));
        }
        let requested_path = Path::new(requested);
        if !requested_path.is_absolute() {
            return Ok(requested_path.to_path_buf());
        }
        let root = proc_fd_path(self.fd.as_raw_fd())
            .map_err(|err| tool_error(tool, ToolErrorKind::Execution, format!("workdir: {err}")))?;
        let wanted = lexical_normalize(requested_path);
        match wanted.strip_prefix(lexical_normalize(&root)) {
            Ok(rel) if rel.as_os_str().is_empty() => Ok(PathBuf::from(".")),
            Ok(rel) => Ok(rel.to_path_buf()),
            Err(_) => Err(escape(tool)),
        }
    }

    pub(crate) fn open_relative(&self, path: &Path, flags: i32) -> std::io::Result<OwnedFd> {
        let root = self.fd.as_raw_fd();
        match openat2(root, path, flags) {
            Ok(fd) => return Ok(fd),
            // Absolute symlinks are rejected even when the target is inside
            // this directory. Rewrite those and open from the directory fd.
            Err(err) if err.raw_os_error() == Some(libc::EXDEV) => {}
            Err(err) => return Err(err),
        }
        open_absolute_symlinks(root, path, flags)
    }
}

impl AsRawFd for WorkDirFd {
    fn as_raw_fd(&self) -> i32 {
        self.fd.as_raw_fd()
    }
}

/// `openat2` with `RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS`. In-jail symlinks
/// are followed. A symlink or `..` that leaves the directory fd is `EXDEV`.
fn openat2(dirfd: i32, path: &Path, flags: i32) -> std::io::Result<OwnedFd> {
    let c_path = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path contains a null byte",
        )
    })?;
    let how = OpenHow {
        flags: u64::from(u32::try_from(flags).unwrap_or(0)),
        mode: 0,
        resolve: libc::RESOLVE_BENEATH | libc::RESOLVE_NO_MAGICLINKS,
    };
    loop {
        let rc = unsafe {
            // SAFETY: `dirfd` is a directory fd we own, `c_path` is NUL-terminated,
            // and `how` matches the 24-byte `open_how` the kernel reads for `size`.
            libc::syscall(
                libc::SYS_openat2,
                dirfd,
                c_path.as_ptr(),
                &how as *const OpenHow,
                std::mem::size_of::<OpenHow>(),
            )
        };
        if rc >= 0 {
            let fd = i32::try_from(rc).map_err(|_| {
                std::io::Error::other("openat2 returned a file descriptor that does not fit in i32")
            })?;
            // SAFETY: non-negative `syscall` result is a new fd owned by the caller.
            return Ok(unsafe { OwnedFd::from_raw_fd(fd) });
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::EINTR) {
            return Err(err);
        }
    }
}

/// Kernel `struct open_how`. libc marks its copy `non_exhaustive`, so the
/// syscall receives this layout-compatible stand-in.
#[repr(C)]
struct OpenHow {
    flags: u64,
    mode: u64,
    resolve: u64,
}

const MAX_SYMLINKS: u32 = 40;

/// `RESOLVE_BENEATH` refused `path`. Follow symlinks with `readlinkat` and
/// open only through directory fds. An absolute link is accepted only when
/// its target is still under `root`.
fn open_absolute_symlinks(root: i32, path: &Path, flags: i32) -> std::io::Result<OwnedFd> {
    let root_lex = lexical_normalize(&proc_fd_path(root)?);
    let mut queue = VecDeque::new();
    push_components(&mut queue, path);
    let mut dir = dup_fd(root)?;
    let mut stack: Vec<OsString> = Vec::new();
    let mut depth = 0u32;

    while let Some(name) = queue.pop_front() {
        if name == ".." {
            if stack.pop().is_none() {
                return Err(std::io::Error::from_raw_os_error(libc::EXDEV));
            }
            dir = reopen_from_root(root, &stack)?;
            continue;
        }
        // `O_DIRECTORY|O_NOFOLLOW` on a symlink is ENOTDIR, which would hide
        // the link. Stat first so an absolute link is judged before open.
        let meta = fstatat_nofollow(dir.as_raw_fd(), &name)?;
        if meta.st_mode & libc::S_IFMT == libc::S_IFLNK {
            depth += 1;
            if depth > MAX_SYMLINKS {
                return Err(std::io::Error::from_raw_os_error(libc::ELOOP));
            }
            let target = readlinkat_path(dir.as_raw_fd(), &name)?;
            let mut spliced = VecDeque::new();
            if target.is_absolute() {
                let lex = lexical_normalize(&target);
                let rel = lex
                    .strip_prefix(&root_lex)
                    .map_err(|_| std::io::Error::from_raw_os_error(libc::EXDEV))?;
                push_components(&mut spliced, rel);
                spliced.append(&mut queue);
                queue = spliced;
                stack.clear();
                dir = dup_fd(root)?;
            } else {
                push_components(&mut spliced, &target);
                spliced.append(&mut queue);
                queue = spliced;
            }
            continue;
        }
        let step_flags = if queue.is_empty() {
            flags
        } else {
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NONBLOCK
        };
        match openat_nofollow(dir.as_raw_fd(), &name, step_flags) {
            Ok(fd) => {
                if queue.is_empty() {
                    return Ok(fd);
                }
                stack.push(name);
                dir = fd;
            }
            // Replaced with a symlink between stat and open. Look again,
            // but stop if the name keeps changing under us.
            Err(err) if err.raw_os_error() == Some(libc::ELOOP) => {
                depth += 1;
                if depth > MAX_SYMLINKS {
                    return Err(err);
                }
                queue.push_front(name);
            }
            Err(err) => return Err(err),
        }
    }
    openat_nofollow(dir.as_raw_fd(), OsStr::new("."), flags)
}

fn push_components(queue: &mut VecDeque<OsString>, path: &Path) {
    for component in path.components() {
        match component {
            Component::Normal(name) => queue.push_back(name.to_os_string()),
            Component::ParentDir => queue.push_back(OsString::from("..")),
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
        }
    }
}

fn reopen_from_root(root: i32, stack: &[OsString]) -> std::io::Result<OwnedFd> {
    let mut current = dup_fd(root)?;
    for name in stack {
        current = openat_nofollow(
            current.as_raw_fd(),
            name,
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NONBLOCK,
        )?;
    }
    Ok(current)
}

fn fstatat_nofollow(dirfd: i32, name: &OsStr) -> std::io::Result<libc::stat> {
    let c_name = CString::new(name.as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path contains a null byte",
        )
    })?;
    loop {
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        let rc = unsafe {
            // SAFETY: `c_name` is NUL-terminated. `AT_SYMLINK_NOFOLLOW` stats the
            // link itself. `st` is writable storage for the result.
            libc::fstatat(
                dirfd,
                c_name.as_ptr(),
                st.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if rc == 0 {
            // SAFETY: `fstatat` returned 0, so it wrote a complete `stat`.
            return Ok(unsafe { st.assume_init() });
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::EINTR) {
            return Err(err);
        }
    }
}

fn openat_nofollow(dirfd: i32, name: &OsStr, flags: i32) -> std::io::Result<OwnedFd> {
    let c_name = CString::new(name.as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path contains a null byte",
        )
    })?;
    let flags = flags | libc::O_NOFOLLOW | libc::O_CLOEXEC;
    loop {
        let fd = unsafe {
            // SAFETY: `dirfd` is a directory fd owned by the caller and `c_name`
            // is a single NUL-terminated component. `O_NOFOLLOW` stops the leaf
            // from being a symlink the caller has not checked.
            libc::openat(dirfd, c_name.as_ptr(), flags)
        };
        if fd >= 0 {
            // SAFETY: non-negative `openat` result is a new fd owned by the caller.
            return Ok(unsafe { OwnedFd::from_raw_fd(fd) });
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::EINTR) {
            return Err(err);
        }
    }
}

fn readlinkat_path(dirfd: i32, name: &OsStr) -> std::io::Result<PathBuf> {
    let c_name = CString::new(name.as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path contains a null byte",
        )
    })?;
    let mut buf = vec![0u8; 256];
    loop {
        let rc = unsafe {
            // SAFETY: `c_name` is NUL-terminated and `buf` is writable for `buf.len()` bytes.
            libc::readlinkat(
                dirfd,
                c_name.as_ptr(),
                buf.as_mut_ptr() as *mut libc::c_char,
                buf.len(),
            )
        };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return Err(err);
        }
        let n = rc as usize;
        if n < buf.len() {
            buf.truncate(n);
            return Ok(PathBuf::from(OsStr::from_bytes(&buf)));
        }
        if buf.len() >= 64 * 1024 {
            return Err(std::io::Error::from_raw_os_error(libc::ENAMETOOLONG));
        }
        buf.resize(buf.len() * 2, 0);
    }
}

fn dup_fd(fd: i32) -> std::io::Result<OwnedFd> {
    let new = unsafe {
        // SAFETY: `fd` is open. `F_DUPFD_CLOEXEC` returns a new descriptor.
        libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0)
    };
    if new < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: non-negative `fcntl` result is a new fd owned by the caller.
    Ok(unsafe { OwnedFd::from_raw_fd(new) })
}

fn proc_fd_path(fd: i32) -> std::io::Result<PathBuf> {
    std::fs::read_link(format!("/proc/self/fd/{fd}"))
}

pub(crate) fn fstat_fd(fd: i32) -> std::io::Result<libc::stat> {
    loop {
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        let rc = unsafe {
            // SAFETY: `fd` is open and `st` is a writable `stat` for `fstat`.
            libc::fstat(fd, st.as_mut_ptr())
        };
        if rc == 0 {
            // SAFETY: `fstat` returned 0, so it wrote a complete `stat`.
            return Ok(unsafe { st.assume_init() });
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::EINTR) {
            return Err(err);
        }
    }
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) => {}
            Component::RootDir => out = PathBuf::from("/"),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(name) => out.push(name),
        }
    }
    out
}

pub(crate) fn escape(tool: &str) -> Error {
    tool_error(
        tool,
        ToolErrorKind::Execution,
        "path escapes the working directory",
    )
}

pub(crate) fn open_error(tool: &str, requested: &str, err: std::io::Error) -> Error {
    // RESOLVE_BENEATH reports an escape as EXDEV. O_NOFOLLOW on a write leaf
    // reports a symlink as ELOOP. Either one is a jail failure.
    match err.raw_os_error() {
        Some(libc::ELOOP) | Some(libc::EXDEV) => escape(tool),
        Some(libc::ENOENT) => tool_error(
            tool,
            ToolErrorKind::Execution,
            format!("path not found: {requested}"),
        ),
        _ => tool_error(tool, ToolErrorKind::Execution, format!("path: {err}")),
    }
}
