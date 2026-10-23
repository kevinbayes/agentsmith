use std::path::Path;

use serde::Deserialize;

use agentsmith_core::Error;

/// How a run drives the model. `Plan` parses so a spec file can name it;
/// this build only executes [`LoopMode::React`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoopMode {
    React,
    Plan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentSpec {
    pub id: String,
    pub name: String,
    pub description: String,
    pub mode: LoopMode,
    pub system_prompt: String,
    pub tools: Option<Vec<String>>,
    /// `None` when the file omitted `model` or set it empty. The caller then
    /// uses `llm.model`.
    pub model: Option<String>,
}

impl AgentSpec {
    /// Spec used when `--spec` is omitted and the caller already selected ReAct.
    pub fn react_default() -> Self {
        Self {
            id: "react".to_string(),
            name: "react".to_string(),
            description: String::new(),
            mode: LoopMode::React,
            system_prompt: "Use the tools you need, then call finish with the final answer."
                .to_string(),
            tools: None,
            model: None,
        }
    }
}

/// Explicit mode wins. Otherwise the spec's mode. Otherwise the caller must say.
pub fn resolve_mode(
    explicit: Option<LoopMode>,
    spec: Option<&AgentSpec>,
) -> Result<LoopMode, Error> {
    if let Some(mode) = explicit {
        return Ok(mode);
    }
    if let Some(spec) = spec {
        return Ok(spec.mode);
    }
    Err(Error::Config(
        "mode is required: pass --mode or JSON mode, or set mode in the spec".into(),
    ))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecFile {
    id: String,
    name: String,
    #[serde(default)]
    description: String,
    mode: LoopMode,
    system_prompt: String,
    #[serde(default)]
    tools: Option<Vec<String>>,
    #[serde(default)]
    model: String,
}

pub fn load_spec(path: &Path) -> Result<AgentSpec, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| Error::Config(format!("reading {}: {err}", path.display())))?;
    parse(&text).map_err(|err| Error::Config(format!("{}: {err}", path.display())))
}

pub fn parse(text: &str) -> Result<AgentSpec, toml::de::Error> {
    let file: SpecFile = toml::from_str(text)?;
    let model = file.model.trim();
    Ok(AgentSpec {
        id: file.id,
        name: file.name,
        description: file.description,
        mode: file.mode,
        system_prompt: file.system_prompt,
        tools: file.tools,
        model: if model.is_empty() {
            None
        } else {
            Some(model.to_string())
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
id = "coder"
name = "Coder"
description = "Implements a task inside the working directory."
mode = "react"
system_prompt = "You are a careful software engineer. Stay inside the working directory."
tools = ["read_file", "list_dir", "write_file", "shell"]
model = ""
"#;

    #[test]
    fn sample_spec_parses_and_empty_model_is_none() {
        let spec = parse(SAMPLE).unwrap();
        assert_eq!(spec.id, "coder");
        assert_eq!(spec.name, "Coder");
        assert_eq!(spec.mode, LoopMode::React);
        assert!(spec.model.is_none());
        assert_eq!(
            spec.tools,
            Some(vec![
                "read_file".into(),
                "list_dir".into(),
                "write_file".into(),
                "shell".into()
            ])
        );
    }

    #[test]
    fn unknown_key_fails_and_plan_mode_parses() {
        let mut text = SAMPLE.to_string();
        text.push_str("role = \"x\"\n");
        assert!(parse(&text).is_err());
        assert!(parse(
            "id = \"a\"\nname = \"A\"\nmode = \"react\"\nsystem_prompt = \"x\"\nsubagent_tools = [\"read_file\"]\n"
        )
        .is_err());
        let plan = parse("id = \"p\"\nname = \"P\"\nmode = \"plan\"\nsystem_prompt = \"x\"\n");
        assert_eq!(plan.unwrap().mode, LoopMode::Plan);
    }

    #[test]
    fn omitted_tools_stay_none_and_empty_list_is_some() {
        let spec =
            parse("id = \"a\"\nname = \"A\"\nmode = \"react\"\nsystem_prompt = \"x\"\n").unwrap();
        assert!(spec.tools.is_none());
        let spec = parse(
            "id = \"a\"\nname = \"A\"\nmode = \"react\"\nsystem_prompt = \"x\"\ntools = []\n",
        )
        .unwrap();
        assert_eq!(spec.tools, Some(Vec::new()));
    }

    #[test]
    fn resolve_mode_prefers_explicit_then_spec() {
        let spec = AgentSpec {
            mode: LoopMode::Plan,
            ..AgentSpec::react_default()
        };
        assert_eq!(
            resolve_mode(Some(LoopMode::React), Some(&spec)).unwrap(),
            LoopMode::React
        );
        assert_eq!(resolve_mode(None, Some(&spec)).unwrap(), LoopMode::Plan);
        assert_eq!(
            resolve_mode(Some(LoopMode::React), None).unwrap(),
            LoopMode::React
        );
        let err = resolve_mode(None, None).unwrap_err();
        assert!(err.to_string().contains("mode is required"), "{err}");
    }
}
