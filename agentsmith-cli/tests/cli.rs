use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_agentsmith"))
}

fn isolated(args: &[&std::ffi::OsStr]) -> std::process::Output {
    let mut cmd = bin();
    // The process may carry AGENTSMITH_* that would override the file under test.
    cmd.env_clear();
    if let Some(path_env) = std::env::var_os("PATH") {
        cmd.env("PATH", path_env);
    }
    cmd.args(args).output().unwrap()
}

fn config_check(path: &std::path::Path) -> std::process::Output {
    isolated(&[
        std::ffi::OsStr::new("--config"),
        path.as_os_str(),
        std::ffi::OsStr::new("config"),
        std::ffi::OsStr::new("check"),
    ])
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

fn write_run_config(dir: &std::path::Path, body: &str) -> std::path::PathBuf {
    let path = dir.join("config.toml");
    std::fs::write(&path, body).unwrap();
    path
}

fn run_output(dir: &std::path::Path, extra: &[&str]) -> std::process::Output {
    let config = dir.join("config.toml");
    let mut args = vec![
        std::ffi::OsString::from("--config"),
        config.as_os_str().to_os_string(),
        std::ffi::OsString::from("run"),
    ];
    for arg in extra {
        args.push(std::ffi::OsString::from(arg));
    }
    let borrowed: Vec<&std::ffi::OsStr> = args.iter().map(std::ffi::OsStr::new).collect();
    isolated(&borrowed)
}

#[test]
fn run_without_mode_exits_2() {
    let dir = scratch();
    write_run_config(
        &dir,
        &format!("harness.data_dir = \"{}\"\n", dir.join("data").display()),
    );
    let out = run_output(&dir, &["--task", "hello"]);
    assert_eq!(
        out.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("mode is required"), "{stderr}");
    assert!(
        out.stdout.is_empty(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn run_spec_plan_mode_exits_2() {
    let dir = scratch();
    write_run_config(
        &dir,
        &format!(
            "[llm]\napi_key = \"k\"\nbase_url = \"http://127.0.0.1:9\"\ntimeout_secs = 1\nretries = 0\n\n[harness]\ndata_dir = \"{}\"\n",
            dir.join("data").display()
        ),
    );
    let spec = dir.join("agent.toml");
    std::fs::write(
        &spec,
        "id = \"coder\"\nname = \"Coder\"\nmode = \"plan\"\nsystem_prompt = \"Plan it.\"\n",
    )
    .unwrap();
    let out = run_output(
        &dir,
        &[
            "--task",
            "hello",
            "--mode",
            "plan",
            "--spec",
            spec.to_str().unwrap(),
        ],
    );
    assert_eq!(
        out.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("llm.model"), "{stderr}");
    assert!(!stderr.contains("not in this build"), "{stderr}");
    assert!(
        out.stdout.is_empty(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn run_mode_react_without_a_model_exits_2() {
    let dir = scratch();
    write_run_config(
        &dir,
        &format!(
            "[llm]\napi_key = \"k\"\nbase_url = \"http://127.0.0.1:9\"\ntimeout_secs = 1\nretries = 0\n\n[harness]\ndata_dir = \"{}\"\n",
            dir.join("data").display()
        ),
    );
    let out = run_output(&dir, &["--task", "hello", "--mode", "react"]);
    assert_eq!(
        out.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("llm.model"), "{stderr}");
    assert!(out.stdout.is_empty());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn run_unknown_spec_tool_exits_2_without_a_transcript() {
    let dir = scratch();
    let data = dir.join("data");
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    write_run_config(
        &dir,
        &format!(
            "[llm]\nmodel = \"m\"\napi_key = \"k\"\nbase_url = \"http://127.0.0.1:9\"\ntimeout_secs = 1\nretries = 0\n\n[harness]\ndata_dir = \"{}\"\nworkdir = \"{}\"\n\n[tools.shell]\nenabled = false\n",
            data.display(),
            work.display()
        ),
    );
    let spec = dir.join("agent.toml");
    std::fs::write(
        &spec,
        "id = \"coder\"\nname = \"Coder\"\nmode = \"react\"\nsystem_prompt = \"Go.\"\ntools = [\"nope\"]\n",
    )
    .unwrap();
    let out = run_output(
        &dir,
        &[
            "--task",
            "hello",
            "--mode",
            "react",
            "--spec",
            spec.to_str().unwrap(),
            "--workdir",
            work.to_str().unwrap(),
        ],
    );
    assert_eq!(
        out.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("nope"), "{stderr}");
    assert!(
        out.stdout.is_empty(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(!data.join("runs").exists());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn run_mode_plan_is_accepted_and_still_needs_an_api_key() {
    let dir = scratch();
    write_run_config(&dir, "llm.model = \"m\"\n");
    let out = run_output(&dir, &["--task", "hello", "--mode", "plan"]);
    assert_eq!(
        out.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("llm.api_key"), "{stderr}");
    assert!(!stderr.contains("not in this build"), "{stderr}");
    assert!(out.stdout.is_empty());
    let _ = std::fs::remove_dir_all(dir);
}
