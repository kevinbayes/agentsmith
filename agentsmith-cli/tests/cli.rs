use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_agentsmith"))
}

fn config_check(path: &std::path::Path) -> std::process::Output {
    let mut cmd = bin();
    // The process may carry AGENTSMITH_* that would override the file under test.
    cmd.env_clear();
    if let Some(path_env) = std::env::var_os("PATH") {
        cmd.env("PATH", path_env);
    }
    cmd.arg("--config")
        .arg(path)
        .arg("config")
        .arg("check")
        .output()
        .unwrap()
}

fn scratch() -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "agentsmith-cli-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn version_prints_the_crate_version() {
    let out = bin().arg("--version").output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains(env!("CARGO_PKG_VERSION")), "{stdout}");
}

#[test]
fn config_check_prints_twelve_hex_chars() {
    let dir = scratch();
    let path = dir.join("ok.toml");
    std::fs::write(&path, "harness.name = \"cli-test\"\n").unwrap();
    let out = config_check(&path);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    let trimmed = stdout.trim_end();
    assert_eq!(stdout, format!("{trimmed}\n"));
    assert_eq!(trimmed.len(), 12);
    assert!(trimmed.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn config_check_bad_file_exits_nonzero_without_a_fingerprint() {
    let dir = scratch();
    let path = dir.join("bad.toml");
    std::fs::write(&path, "[loop]\nmax_fanout = 9\n").unwrap();
    let out = config_check(&path);
    assert!(!out.status.success());
    assert!(
        out.stdout.is_empty(),
        "stdout: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(!String::from_utf8_lossy(&out.stderr).is_empty());
    let _ = std::fs::remove_dir_all(dir);
}
