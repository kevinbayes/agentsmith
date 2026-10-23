//! systemd user and system units. Non-Linux is a flag so tests can take the
//! refusal path without pretending the host OS changed.

use std::ffi::CString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use agentsmith_core::redact::redact;
use agentsmith_core::Error;

pub const LINUX_ONLY: &str = "systemd service management is Linux-only in v1";
const UNIT_NAME: &str = "agentsmith.service";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    User,
    System,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub scope: Scope,
    pub unit_path: PathBuf,
    pub config_path: PathBuf,
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
}

impl Layout {
    pub fn config_mode(&self) -> u32 {
        match self.scope {
            Scope::User => 0o600,
            Scope::System => 0o640,
        }
    }

    pub fn data_mode(&self) -> u32 {
        match self.scope {
            Scope::User => 0o700,
            Scope::System => 0o750,
        }
    }
}

pub fn layout(scope: Scope, home: &Path) -> Layout {
    match scope {
        Scope::User => Layout {
            scope,
            unit_path: home.join(".config/systemd/user").join(UNIT_NAME),
            config_path: home.join(".config/agentsmith/config.toml"),
            config_dir: home.join(".config/agentsmith"),
            data_dir: home.join(".local/share/agentsmith"),
        },
        Scope::System => Layout {
            scope,
            unit_path: PathBuf::from("/etc/systemd/system").join(UNIT_NAME),
            config_path: PathBuf::from("/etc/agentsmith/config.toml"),
            config_dir: PathBuf::from("/etc/agentsmith"),
            data_dir: PathBuf::from("/var/lib/agentsmith"),
        },
    }
}

#[derive(Clone, Debug)]
pub struct UnitSpec<'a> {
    pub system: bool,
    pub exe: &'a Path,
    pub config: &'a Path,
    pub data_dir: &'a Path,
    pub config_dir: &'a Path,
}

pub fn render_unit(spec: &UnitSpec<'_>) -> String {
    let account = if spec.system {
        "User=agentsmith\nGroup=agentsmith\n"
    } else {
        ""
    };
    let wanted = if spec.system {
        "multi-user.target"
    } else {
        "default.target"
    };
    format!(
        "\
[Unit]
Description=AgentSmith service
After=network-online.target
Wants=network-online.target
# ProtectHome=read-only breaks a system service whose workdir is under /home. The system unit is for the agentsmith user. Developers use the user unit.

[Service]
Type=simple
ExecStart={exe} run --service --config {config}
WorkingDirectory={data}
Environment=RUST_LOG=info
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
ProtectSystem=full
ProtectHome=read-only
ReadWritePaths={data} {config_dir}
{account}\
[Install]
WantedBy={wanted}
",
        exe = unit_arg(spec.exe),
        config = unit_arg(spec.config),
        data = unit_arg(spec.data_dir),
        config_dir = unit_arg(spec.config_dir),
        account = account,
        wanted = wanted,
    )
}

fn unit_arg(path: &Path) -> String {
    let text = path.display().to_string();
    if text.chars().any(|ch| ch.is_whitespace()) {
        format!("\"{}\"", text.replace('"', "\\\""))
    } else {
        text
    }
}

pub fn useradd_argv() -> Vec<String> {
    vec![
        "useradd".into(),
        "--system".into(),
        "--home-dir".into(),
        "/var/lib/agentsmith".into(),
        "--shell".into(),
        "/usr/sbin/nologin".into(),
        "agentsmith".into(),
    ]
}

fn systemctl_base(scope: Scope) -> Vec<String> {
    match scope {
        Scope::User => vec!["systemctl".into(), "--user".into()],
        Scope::System => vec!["systemctl".into()],
    }
}

/// `useradd` (system, user missing) then `daemon-reload`, `enable`, and
/// `restart` unless `no_start`.
pub fn install_argvs(scope: Scope, no_start: bool, user_missing: bool) -> Vec<Vec<String>> {
    let mut commands = Vec::new();
    if scope == Scope::System && user_missing {
        commands.push(useradd_argv());
    }
    commands.extend(systemctl_argvs(scope, no_start));
    commands
}

fn systemctl_argvs(scope: Scope, no_start: bool) -> Vec<Vec<String>> {
    let base = systemctl_base(scope);
    let mut reload = base.clone();
    reload.push("daemon-reload".into());
    let mut enable = base.clone();
    enable.push("enable".into());
    enable.push(UNIT_NAME.into());
    let mut commands = vec![reload, enable];
    if !no_start {
        let mut restart = base;
        restart.push("restart".into());
        restart.push(UNIT_NAME.into());
        commands.push(restart);
    }
    commands
}

/// `disable --now`, then `daemon-reload`. The unit file is removed between
/// the two reloads. The returned list is the three systemctl invocations.
pub fn uninstall_argvs(scope: Scope) -> Vec<Vec<String>> {
    let base = systemctl_base(scope);
    let mut disable = base.clone();
    disable.push("disable".into());
    disable.push("--now".into());
    disable.push(UNIT_NAME.into());
    let mut reload = base;
    reload.push("daemon-reload".into());
    vec![disable, reload.clone(), reload]
}

/// Exit 0, or a non-zero exit whose output says the unit is already gone.
pub fn disable_succeeded(code: i32, output: &str) -> bool {
    if code == 0 {
        return true;
    }
    let lower = output.to_ascii_lowercase();
    lower.contains("not loaded") || lower.contains("does not exist")
}

#[derive(Clone, Debug)]
pub struct ExecOutput {
    pub code: i32,
    pub output: String,
}

pub struct InstallRequest {
    pub linux: bool,
    pub scope: Scope,
    pub no_start: bool,
    pub user_missing: bool,
    pub home: PathBuf,
    pub exe: PathBuf,
    pub layout_override: Option<Layout>,
}

pub fn perform_install(
    request: &InstallRequest,
    stdout: &mut dyn Write,
    mut exec: impl FnMut(&[String]) -> ExecOutput,
) -> Result<Option<String>, Error> {
    if !request.linux {
        return Err(Error::Config(LINUX_ONLY.into()));
    }
    let paths = request
        .layout_override
        .clone()
        .unwrap_or_else(|| layout(request.scope, &request.home));
    // useradd before any seed. systemctl only after the unit and config exist.
    let argvs = install_argvs(request.scope, request.no_start, request.user_missing);
    let systemctl_at = if request.scope == Scope::System && request.user_missing {
        let output = exec(&argvs[0]);
        if output.code != 0 {
            return Err(command_failed(&argvs[0], &output));
        }
        1
    } else {
        0
    };
    write_unit(&paths, &request.exe)?;
    ensure_dir(&paths.data_dir, paths.data_mode())?;
    if paths.scope == Scope::System {
        chown_agentsmith(&paths.data_dir)?;
    }
    let token = seed_if_absent(&paths)?;
    for argv in &argvs[systemctl_at..] {
        let output = exec(argv);
        if output.code != 0 {
            return Err(command_failed(argv, &output));
        }
    }
    if let Some(token) = &token {
        writeln!(stdout, "{token}")?;
    }
    if request.scope == Scope::User {
        linger_hint(stdout)?;
    }
    Ok(token)
}

pub fn perform_uninstall(
    linux: bool,
    scope: Scope,
    purge: bool,
    paths: &Layout,
    mut exec: impl FnMut(&[String]) -> ExecOutput,
) -> Result<(), Error> {
    if !linux {
        return Err(Error::Config(LINUX_ONLY.into()));
    }
    let argvs = uninstall_argvs(scope);
    let disabled = exec(&argvs[0]);
    if !disable_succeeded(disabled.code, &disabled.output) {
        return Err(command_failed(&argvs[0], &disabled));
    }
    let reloaded = exec(&argvs[1]);
    if reloaded.code != 0 {
        return Err(command_failed(&argvs[1], &reloaded));
    }
    match fs::remove_file(&paths.unit_path) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }
    let reloaded = exec(&argvs[2]);
    if reloaded.code != 0 {
        return Err(command_failed(&argvs[2], &reloaded));
    }
    if purge {
        match fs::remove_dir_all(&paths.data_dir) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

pub fn execute_install(system: bool, no_start: bool) -> Result<(), Error> {
    let home = home_dir()?;
    let exe = current_exe();
    let scope = if system { Scope::System } else { Scope::User };
    let user_missing = system && !agentsmith_user_exists();
    let mut stdout = std::io::stdout();
    perform_install(
        &InstallRequest {
            linux: cfg!(target_os = "linux"),
            scope,
            no_start,
            user_missing,
            home,
            exe,
            layout_override: None,
        },
        &mut stdout,
        run_exec,
    )?;
    Ok(())
}

pub fn execute_uninstall(system: bool, purge: bool) -> Result<(), Error> {
    let home = home_dir()?;
    let scope = if system { Scope::System } else { Scope::User };
    let paths = layout(scope, &home);
    perform_uninstall(cfg!(target_os = "linux"), scope, purge, &paths, run_exec)
}

fn write_unit(paths: &Layout, exe: &Path) -> Result<(), Error> {
    if let Some(parent) = paths.unit_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = render_unit(&UnitSpec {
        system: paths.scope == Scope::System,
        exe,
        config: &paths.config_path,
        data_dir: &paths.data_dir,
        config_dir: &paths.config_dir,
    });
    fs::write(&paths.unit_path, body)?;
    Ok(())
}

fn seed_if_absent(paths: &Layout) -> Result<Option<String>, Error> {
    if paths.config_path.exists() {
        return Ok(None);
    }
    let token = generate_token()?;
    let wrote = write_seed_file(
        &paths.config_path,
        &paths.data_dir,
        &token,
        paths.config_mode(),
    )?;
    if !wrote {
        return Ok(None);
    }
    if paths.scope == Scope::System {
        chown_agentsmith(&paths.config_path)?;
    }
    Ok(Some(token))
}

/// Write the seed only when `token` is a real install token. An empty token
/// is refused before the file is created.
pub fn write_seed_file(
    path: &Path,
    data_dir: &Path,
    token: &str,
    mode: u32,
) -> Result<bool, Error> {
    if !valid_token(token) {
        return Err(Error::Config(
            "refusing to write an empty service token".into(),
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => return Ok(false),
        Err(err) => return Err(err.into()),
    };
    let body = render_seed(data_dir, token);
    if let Err(err) = file.write_all(body.as_bytes()) {
        drop(file);
        let _ = fs::remove_file(path);
        return Err(err.into());
    }
    let mut perms = file.metadata()?.permissions();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        perms.set_mode(mode);
    }
    let _ = mode;
    file.set_permissions(perms)?;
    Ok(true)
}

fn render_seed(data_dir: &Path, token: &str) -> String {
    format!(
        "\
[harness]
data_dir = {data}
log_level = \"info\"

[service]
bind = \"127.0.0.1:7433\"
max_runs = 2
token = {token}

[reporter]
enabled = false
",
        data = toml_string(&data_dir.display().to_string()),
        token = toml_string(token),
    )
}

fn toml_string(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

fn valid_token(token: &str) -> bool {
    token.len() == 64
        && token
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn generate_token() -> Result<String, Error> {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes)
        .map_err(|err| Error::Io(std::io::Error::other(err.to_string())))?;
    let token = hex_encode(&bytes);
    if !valid_token(&token) {
        return Err(Error::Config(
            "refusing to write an empty service token".into(),
        ));
    }
    Ok(token)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

fn ensure_dir(path: &Path, mode: u32) -> Result<(), Error> {
    if path.is_dir() {
        return Ok(());
    }
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(path)?.permissions();
        perms.set_mode(mode);
        fs::set_permissions(path, perms)?;
    }
    let _ = mode;
    Ok(())
}

/// `chown` is a no-op when the account is absent or the process is not root.
fn chown_agentsmith(path: &Path) -> Result<(), Error> {
    if unsafe { libc::geteuid() } != 0 {
        return Ok(());
    }
    let Some((uid, gid)) = agentsmith_ids() else {
        return Ok(());
    };
    std::os::unix::fs::chown(path, Some(uid), Some(gid))?;
    Ok(())
}

pub fn agentsmith_user_exists() -> bool {
    agentsmith_ids().is_some()
}

fn agentsmith_ids() -> Option<(u32, u32)> {
    let name = CString::new("agentsmith").expect("static name");
    // The pointer is only read for uid and gid before the next libc call.
    let pwd = unsafe { libc::getpwnam(name.as_ptr()) };
    if pwd.is_null() {
        return None;
    }
    let pwd = unsafe { &*pwd };
    Some((pwd.pw_uid, pwd.pw_gid))
}

fn linger_hint(stdout: &mut dyn Write) -> Result<(), Error> {
    let user = std::env::var("USER").unwrap_or_default();
    let path = Path::new("/var/lib/systemd/linger").join(&user);
    if !path.exists() {
        writeln!(
            stdout,
            "linger is not enabled, user services stop at logout, loginctl enable-linger {user}"
        )?;
    }
    Ok(())
}

fn home_dir() -> Result<PathBuf, Error> {
    std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| Error::Config("HOME is not set".into()))
}

fn current_exe() -> PathBuf {
    std::env::current_exe().unwrap_or_else(|_| PathBuf::from("agentsmith"))
}

fn run_exec(argv: &[String]) -> ExecOutput {
    if argv.is_empty() {
        return ExecOutput {
            code: 1,
            output: "empty argv".into(),
        };
    }
    match Command::new(&argv[0]).args(&argv[1..]).output() {
        Ok(output) => {
            let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            ExecOutput {
                code: output.status.code().unwrap_or(1),
                output: text,
            }
        }
        Err(err) => ExecOutput {
            code: 1,
            output: err.to_string(),
        },
    }
}

fn command_failed(argv: &[String], output: &ExecOutput) -> Error {
    Error::Io(std::io::Error::other(format!(
        "{} exited {}: {}",
        argv.join(" "),
        output.code,
        redact(&output.output)
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn scratch() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "agentsmith-install-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn temp_layout(root: &Path, scope: Scope) -> Layout {
        Layout {
            scope,
            unit_path: root.join("unit").join(UNIT_NAME),
            config_path: root.join("config").join("config.toml"),
            config_dir: root.join("config"),
            data_dir: root.join("data"),
        }
    }

    fn ok_exec(
        calls: &std::sync::Mutex<Vec<Vec<String>>>,
    ) -> impl FnMut(&[String]) -> ExecOutput + '_ {
        |argv: &[String]| {
            calls.lock().unwrap().push(argv.to_vec());
            ExecOutput {
                code: 0,
                output: String::new(),
            }
        }
    }

    #[test]
    fn units_carry_hardening_and_protect_home_comment() {
        let exe = Path::new("/usr/local/bin/agentsmith");
        let user = layout(Scope::User, Path::new("/home/dev"));
        let system = layout(Scope::System, Path::new("/home/dev"));
        let user_unit = render_unit(&UnitSpec {
            system: false,
            exe,
            config: &user.config_path,
            data_dir: &user.data_dir,
            config_dir: &user.config_dir,
        });
        let system_unit = render_unit(&UnitSpec {
            system: true,
            exe,
            config: &system.config_path,
            data_dir: &system.data_dir,
            config_dir: &system.config_dir,
        });
        for unit in [&user_unit, &system_unit] {
            for key in [
                "After=network-online.target",
                "Wants=network-online.target",
                "Restart=on-failure",
                "RestartSec=5",
                "NoNewPrivileges=true",
                "ProtectSystem=full",
                "ProtectHome=read-only",
                "Environment=RUST_LOG=info",
                "Type=simple",
            ] {
                assert!(
                    unit.lines().any(|line| line == key),
                    "missing {key} in {unit}"
                );
            }
            assert!(unit.contains("run --service --config"), "{unit}");
            assert!(
                unit.contains(&format!("WorkingDirectory={}", user.data_dir.display()))
                    || unit.contains(&format!("WorkingDirectory={}", system.data_dir.display()))
            );
            assert!(unit.contains("ReadWritePaths="));
            assert!(
                unit.lines().any(|line| {
                    line.starts_with('#')
                        && line.contains("ProtectHome=read-only")
                        && line.contains("/home")
                }),
                "{unit}"
            );
        }
        assert!(!user_unit
            .lines()
            .any(|line| line.starts_with("User=") || line.starts_with("Group=")));
        assert!(user_unit.contains("WantedBy=default.target"));
        assert!(user_unit.contains("ExecStart=/usr/local/bin/agentsmith run --service --config /home/dev/.config/agentsmith/config.toml"));
        assert!(user_unit.contains("WorkingDirectory=/home/dev/.local/share/agentsmith"));
        assert!(user_unit.contains(
            "ReadWritePaths=/home/dev/.local/share/agentsmith /home/dev/.config/agentsmith"
        ));
        assert!(system_unit.lines().any(|line| line == "User=agentsmith"));
        assert!(system_unit.lines().any(|line| line == "Group=agentsmith"));
        assert!(system_unit.contains("WantedBy=multi-user.target"));
        assert!(system_unit.contains("ExecStart=/usr/local/bin/agentsmith run --service --config /etc/agentsmith/config.toml"));
        assert!(system_unit.contains("WorkingDirectory=/var/lib/agentsmith"));
        assert!(system_unit.contains("ReadWritePaths=/var/lib/agentsmith /etc/agentsmith"));
    }

    #[test]
    fn install_argv_matches_scope_and_no_start() {
        let user = install_argvs(Scope::User, false, true);
        assert_eq!(
            user,
            vec![
                vec!["systemctl", "--user", "daemon-reload"],
                vec!["systemctl", "--user", "enable", "agentsmith.service"],
                vec!["systemctl", "--user", "restart", "agentsmith.service"],
            ]
        );
        assert!(user
            .iter()
            .all(|argv| argv.iter().any(|arg| arg == "--user")));
        let no_start = install_argvs(Scope::User, true, false);
        assert_eq!(no_start.len(), 2);
        assert!(no_start
            .iter()
            .all(|argv| !argv.iter().any(|arg| arg == "restart")));
        let missing = install_argvs(Scope::System, false, true);
        assert_eq!(missing[0], useradd_argv());
        assert_eq!(
            missing[0],
            [
                "useradd",
                "--system",
                "--home-dir",
                "/var/lib/agentsmith",
                "--shell",
                "/usr/sbin/nologin",
                "agentsmith",
            ]
        );
        assert!(missing
            .iter()
            .skip(1)
            .all(|argv| !argv.iter().any(|arg| arg == "--user")));
        let rest: Vec<Vec<String>> = vec![
            vec!["systemctl".into(), "daemon-reload".into()],
            vec![
                "systemctl".into(),
                "enable".into(),
                "agentsmith.service".into(),
            ],
            vec![
                "systemctl".into(),
                "restart".into(),
                "agentsmith.service".into(),
            ],
        ];
        assert_eq!(&missing[1..], rest.as_slice());
        let present = install_argvs(Scope::System, false, false);
        assert_eq!(present[0][0], "systemctl");
        assert!(!present.iter().any(|argv| argv[0] == "useradd"));
    }

    #[test]
    fn uninstall_argv_and_classifier() {
        let user = uninstall_argvs(Scope::User);
        assert_eq!(
            user,
            vec![
                vec![
                    "systemctl",
                    "--user",
                    "disable",
                    "--now",
                    "agentsmith.service"
                ],
                vec!["systemctl", "--user", "daemon-reload"],
                vec!["systemctl", "--user", "daemon-reload"],
            ]
        );
        let system = uninstall_argvs(Scope::System);
        assert_eq!(
            system[0],
            vec!["systemctl", "disable", "--now", "agentsmith.service"]
        );
        assert!(system
            .iter()
            .all(|argv| !argv.iter().any(|arg| arg == "--user")));
        assert!(disable_succeeded(0, ""));
        assert!(disable_succeeded(1, "Unit agentsmith.service not loaded."));
        assert!(disable_succeeded(5, "Unit DOES NOT EXIST"));
        assert!(!disable_succeeded(1, "permission denied"));
        assert!(!disable_succeeded(1, ""));
    }

    #[test]
    fn failed_disable_does_not_touch_the_unit_or_data_dir() {
        let root = scratch();
        let paths = temp_layout(&root, Scope::User);
        fs::create_dir_all(&paths.data_dir).unwrap();
        fs::create_dir_all(paths.unit_path.parent().unwrap()).unwrap();
        fs::write(&paths.unit_path, "unit").unwrap();
        fs::write(paths.data_dir.join("keep"), "x").unwrap();
        let calls = std::sync::Mutex::new(Vec::new());
        let err = perform_uninstall(true, Scope::User, true, &paths, |argv| {
            calls.lock().unwrap().push(argv.to_vec());
            ExecOutput {
                code: 1,
                output: "permission denied".into(),
            }
        })
        .unwrap_err();
        assert!(err.to_string().contains("permission denied"), "{err}");
        assert_eq!(calls.lock().unwrap().len(), 1);
        assert!(paths.unit_path.is_file());
        assert!(paths.data_dir.join("keep").is_file());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn successful_purge_removes_data_after_both_reloads() {
        let root = scratch();
        let paths = temp_layout(&root, Scope::System);
        fs::create_dir_all(&paths.data_dir).unwrap();
        fs::create_dir_all(paths.unit_path.parent().unwrap()).unwrap();
        fs::write(&paths.unit_path, "unit").unwrap();
        fs::write(paths.config_path.parent().unwrap().join("nope"), "stay").ok();
        fs::create_dir_all(&paths.config_dir).unwrap();
        fs::write(&paths.config_path, "token = \"abc\"\n").unwrap();
        let calls = std::sync::Mutex::new(Vec::new());
        perform_uninstall(true, Scope::System, true, &paths, ok_exec(&calls)).unwrap();
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            uninstall_argvs(Scope::System)
        );
        assert!(!paths.unit_path.exists());
        assert!(!paths.data_dir.exists());
        assert!(paths.config_path.is_file(), "config is not purged");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn not_loaded_disable_still_removes_a_missing_unit() {
        let root = scratch();
        let paths = temp_layout(&root, Scope::User);
        let calls = std::sync::Mutex::new(0u32);
        perform_uninstall(true, Scope::User, true, &paths, |argv| {
            *calls.lock().unwrap() += 1;
            if argv.iter().any(|arg| arg == "disable") {
                ExecOutput {
                    code: 1,
                    output: "Unit agentsmith.service not loaded.".into(),
                }
            } else {
                ExecOutput {
                    code: 0,
                    output: String::new(),
                }
            }
        })
        .unwrap();
        assert_eq!(*calls.lock().unwrap(), 3);
        assert!(!paths.data_dir.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn seed_writes_one_token_and_does_not_overwrite() {
        let root = scratch();
        let paths = temp_layout(&root, Scope::User);
        let calls = std::sync::Mutex::new(Vec::new());
        let mut stdout = Vec::new();
        let token = perform_install(
            &InstallRequest {
                linux: true,
                scope: Scope::User,
                no_start: false,
                user_missing: false,
                home: root.clone(),
                exe: PathBuf::from("/usr/local/bin/agentsmith"),
                layout_override: Some(paths.clone()),
            },
            &mut stdout,
            ok_exec(&calls),
        )
        .unwrap()
        .unwrap();
        assert_eq!(token.len(), 64);
        assert!(token.chars().all(|ch| matches!(ch, '0'..='9' | 'a'..='f')));
        let text = String::from_utf8(stdout.clone()).unwrap();
        assert!(text.lines().any(|line| line == token), "{text}");
        let file = fs::read_to_string(&paths.config_path).unwrap();
        assert!(file.contains(&format!("token = \"{token}\"")), "{file}");
        assert!(file.contains(&format!("data_dir = \"{}\"", paths.data_dir.display())));
        assert!(file.contains("log_level = \"info\""));
        assert!(file.contains("bind = \"127.0.0.1:7433\""));
        assert!(file.contains("max_runs = 2"));
        assert!(file.contains("enabled = false"));
        assert!(!file.contains("[tools]"));
        assert!(!file.contains("endpoint"));
        let mode = fs::metadata(&paths.config_path)
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);
        let data_mode = fs::metadata(&paths.data_dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(data_mode, 0o700);
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            install_argvs(Scope::User, false, false)
        );

        let mut again = Vec::new();
        let second = perform_install(
            &InstallRequest {
                linux: true,
                scope: Scope::User,
                no_start: true,
                user_missing: false,
                home: root.clone(),
                exe: PathBuf::from("/usr/local/bin/agentsmith"),
                layout_override: Some(paths.clone()),
            },
            &mut again,
            ok_exec(&calls),
        )
        .unwrap();
        assert!(second.is_none());
        let again_text = String::from_utf8(again).unwrap();
        assert!(!again_text.lines().any(|line| line == token));
        assert_eq!(fs::read_to_string(&paths.config_path).unwrap(), file);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn empty_token_is_not_written() {
        let root = scratch();
        let path = root.join("config.toml");
        let err = write_seed_file(&path, &root.join("data"), "", 0o600).unwrap_err();
        assert!(err.to_string().contains("empty service token"), "{err}");
        assert!(!path.exists());
        let err = write_seed_file(&path, &root.join("data"), &"A".repeat(64), 0o600).unwrap_err();
        assert!(!path.exists(), "{err}");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn system_seed_mode_is_0640() {
        let root = scratch();
        let path = root.join("config.toml");
        write_seed_file(&path, &root.join("data"), &"ab".repeat(32), 0o640).unwrap();
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o640);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn no_start_omits_restart_and_system_useradd_is_conditional() {
        let root = scratch();
        let paths = temp_layout(&root, Scope::System);
        let calls = std::sync::Mutex::new(Vec::new());
        let mut stdout = Vec::new();
        perform_install(
            &InstallRequest {
                linux: true,
                scope: Scope::System,
                no_start: true,
                user_missing: true,
                home: root.clone(),
                exe: PathBuf::from("/usr/bin/agentsmith"),
                layout_override: Some(paths.clone()),
            },
            &mut stdout,
            ok_exec(&calls),
        )
        .unwrap();
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            install_argvs(Scope::System, true, true)
        );
        assert_eq!(calls.lock().unwrap()[0], useradd_argv());
        let calls = std::sync::Mutex::new(Vec::new());
        perform_install(
            &InstallRequest {
                linux: true,
                scope: Scope::System,
                no_start: false,
                user_missing: false,
                home: root.clone(),
                exe: PathBuf::from("/usr/bin/agentsmith"),
                layout_override: Some(paths),
            },
            &mut Vec::new(),
            ok_exec(&calls),
        )
        .unwrap();
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            install_argvs(Scope::System, false, false)
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn useradd_failure_does_not_write_or_continue() {
        let root = scratch();
        let paths = temp_layout(&root, Scope::System);
        let calls = std::sync::Mutex::new(Vec::new());
        let err = perform_install(
            &InstallRequest {
                linux: true,
                scope: Scope::System,
                no_start: false,
                user_missing: true,
                home: root.clone(),
                exe: PathBuf::from("/usr/bin/agentsmith"),
                layout_override: Some(paths.clone()),
            },
            &mut Vec::new(),
            |argv| {
                calls.lock().unwrap().push(argv.to_vec());
                ExecOutput {
                    code: 1,
                    output: "useradd failed".into(),
                }
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("useradd"), "{err}");
        assert_eq!(calls.lock().unwrap().len(), 1);
        assert!(!paths.unit_path.exists());
        assert!(!paths.config_path.exists());
        assert!(!paths.data_dir.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn non_linux_install_and_uninstall_write_nothing() {
        let root = scratch();
        let paths = temp_layout(&root, Scope::User);
        let err = perform_install(
            &InstallRequest {
                linux: false,
                scope: Scope::User,
                no_start: false,
                user_missing: false,
                home: root.clone(),
                exe: PathBuf::from("/usr/bin/agentsmith"),
                layout_override: Some(paths.clone()),
            },
            &mut Vec::new(),
            |_| panic!("exec on non-linux"),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), format!("config: {LINUX_ONLY}"));
        assert!(err.to_string().contains(LINUX_ONLY));
        let err =
            perform_uninstall(false, Scope::User, true, &paths, |_| panic!("exec")).unwrap_err();
        assert!(err.to_string().contains(LINUX_ONLY), "{err}");
        let entries: Vec<_> = fs::read_dir(&root).unwrap().collect();
        assert!(entries.is_empty(), "{entries:?}");
        let _ = fs::remove_dir_all(root);
    }
}
