use std::collections::BTreeSet;
use std::sync::Arc;

use agentsmith_core::{Error, ToolErrorKind};
use agentsmith_tools::{Tool, ToolCtx, ToolDefinition, ToolOutput, ToolRegistry};

/// Who is asking. This build only constructs [`RunKind::Root`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunKind {
    Root,
}

/// Per-run view. `get` and `call` both refuse a name outside `visible`.
#[derive(Clone)]
pub struct RunTools {
    registry: ToolRegistry,
    visible: BTreeSet<String>,
}

impl RunTools {
    /// Allow-list steps: registered names, optional `tools.allow`, optional
    /// non-empty `spec.tools`, then kind `Root` drops `spawn_subagents`.
    ///
    /// An empty `spec.tools` does not cut. An empty `tools.allow` does.
    pub fn resolve(
        registry: ToolRegistry,
        allow: Option<&[String]>,
        spec_tools: Option<&[String]>,
        kind: RunKind,
    ) -> Result<Self, Error> {
        let registered: BTreeSet<String> = registry.names().into_iter().collect();
        let mut visible = apply_allow(&registered, allow)?;
        apply_spec(&mut visible, spec_tools)?;
        if kind == RunKind::Root {
            visible.remove("spawn_subagents");
        }
        Ok(Self { registry, visible })
    }

    /// Visible tools only, ordered by name.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.visible
            .iter()
            .filter_map(|name| self.registry.get(name).map(|tool| tool.definition()))
            .collect()
    }

    /// `None` when `name` is not in the visible set, even if the process map has it.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        if !self.visible.contains(name) {
            return None;
        }
        self.registry.get(name)
    }

    /// Second check of the visible set, then the implementation.
    pub async fn call(
        &self,
        name: &str,
        ctx: ToolCtx,
        arguments: serde_json::Value,
    ) -> Result<ToolOutput, Error> {
        let Some(tool) = self.get(name) else {
            return Err(Error::Tool {
                tool: name.to_string(),
                kind: ToolErrorKind::NotFound,
                message: format!("no such tool: {name}"),
            });
        };
        tool.call(ctx, arguments).await
    }
}

fn apply_allow(
    registered: &BTreeSet<String>,
    allow: Option<&[String]>,
) -> Result<BTreeSet<String>, Error> {
    let Some(names) = allow else {
        return Ok(registered.clone());
    };
    for name in names {
        if !registered.contains(name) {
            return Err(Error::Config(format!(
                "tools.allow includes \"{name}\", which is not registered"
            )));
        }
    }
    Ok(names.iter().cloned().collect())
}

fn apply_spec(visible: &mut BTreeSet<String>, spec_tools: Option<&[String]>) -> Result<(), Error> {
    let Some(names) = spec_tools else {
        return Ok(());
    };
    // Omitted and empty both mean the spec does not narrow the view.
    if names.is_empty() {
        return Ok(());
    }
    for name in names {
        if !visible.contains(name) {
            return Err(Error::Config(format!(
                "spec.tools includes \"{name}\", which is not available"
            )));
        }
    }
    visible.retain(|name| names.iter().any(|wanted| wanted == name));
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use agentsmith_tools::{ToolCtx, ToolOutput};
    use serde_json::{json, Value};

    use super::*;

    struct Named {
        name: &'static str,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl Tool for Named {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.to_string(),
                description: self.name.to_string(),
                parameters: json!({"type": "object", "additionalProperties": false}),
            }
        }

        async fn call(&self, _ctx: ToolCtx, _arguments: Value) -> Result<ToolOutput, Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutput {
                content: self.name.to_string(),
            })
        }
    }

    fn must_err(result: Result<RunTools, Error>) -> Error {
        match result {
            Err(err) => err,
            Ok(_) => panic!("expected allow-list error"),
        }
    }

    fn registry(names: &[&'static str]) -> (ToolRegistry, Vec<Arc<AtomicUsize>>) {
        let registry = ToolRegistry::new();
        let mut calls = Vec::new();
        for name in names {
            let counter = Arc::new(AtomicUsize::new(0));
            registry
                .register(Arc::new(Named {
                    name,
                    calls: Arc::clone(&counter),
                }))
                .unwrap();
            calls.push(counter);
        }
        (registry, calls)
    }

    #[test]
    fn get_hides_a_registered_tool_and_definitions_are_ordered() {
        let (registry, _) = registry(&["zebra", "hidden", "alpha"]);
        let tools = RunTools::resolve(
            registry,
            None,
            Some(&["zebra".into(), "alpha".into()]),
            RunKind::Root,
        )
        .unwrap();
        assert!(tools.get("hidden").is_none());
        assert!(tools.get("alpha").is_some());
        assert!(tools.get("zebra").is_some());
        let names: Vec<_> = tools
            .definitions()
            .into_iter()
            .map(|def| def.name)
            .collect();
        assert_eq!(names, ["alpha", "zebra"]);
    }

    #[test]
    fn allow_and_spec_fail_closed_and_empty_spec_does_not_cut() {
        let (registry, _) = registry(&["alpha", "zebra"]);
        let err = must_err(RunTools::resolve(
            registry.clone(),
            Some(&["alpha".into(), "missing".into()]),
            None,
            RunKind::Root,
        ));
        assert!(err.to_string().contains("missing"), "{err}");

        let err = must_err(RunTools::resolve(
            registry.clone(),
            Some(&["alpha".into()]),
            Some(&["zebra".into()]),
            RunKind::Root,
        ));
        assert!(err.to_string().contains("zebra"), "{err}");

        let tools = RunTools::resolve(registry.clone(), Some(&[]), None, RunKind::Root).unwrap();
        assert!(tools.definitions().is_empty());

        let tools = RunTools::resolve(registry, None, Some(&[]), RunKind::Root).unwrap();
        assert_eq!(tools.definitions().len(), 2);
    }

    #[test]
    fn root_drops_spawn_subagents_after_the_intersection() {
        let (registry, calls) = registry(&["read_file", "spawn_subagents"]);
        let tools = RunTools::resolve(registry, None, None, RunKind::Root).unwrap();
        assert!(tools.get("spawn_subagents").is_none());
        assert!(tools.get("read_file").is_some());
        assert_eq!(calls[1].load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn call_refuses_a_name_that_is_registered_but_hidden() {
        let (registry, calls) = registry(&["alpha", "hidden"]);
        let tools =
            RunTools::resolve(registry, None, Some(&["alpha".into()]), RunKind::Root).unwrap();
        let err = tools
            .call(
                "hidden",
                ToolCtx {
                    run_id: uuid::Uuid::nil(),
                    workdir: std::env::temp_dir(),
                    cancel: agentsmith_tools::CancellationToken::new(),
                },
                json!({}),
            )
            .await
            .unwrap_err();
        match err {
            Error::Tool { kind, message, .. } => {
                assert_eq!(kind, ToolErrorKind::NotFound);
                assert_eq!(message, "no such tool: hidden");
            }
            other => panic!("unexpected {other}"),
        }
        assert_eq!(calls[1].load(Ordering::SeqCst), 0);
    }
}
