use std::collections::BTreeSet;
use std::sync::Arc;

use agentsmith_core::{Error, ToolErrorKind};
use agentsmith_tools::{Tool, ToolCtx, ToolDefinition, ToolOutput, ToolRegistry};

pub const SPAWN_SUBAGENTS: &str = "spawn_subagents";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunKind {
    Root,
    Step,
    Spawned,
}

/// Inputs for the four allow-list steps.
pub struct ToolFilter<'a> {
    pub allow: Option<&'a [String]>,
    pub spec_tools: Option<&'a [String]>,
    /// Applied only for [`RunKind::Spawned`]. `None` inherits the parent view.
    pub subagent_tools: Option<&'a [String]>,
    pub kind: RunKind,
    pub depth: u32,
    pub max_subagent_depth: u32,
}

/// Per-run filter. `definitions` and `get` apply it to the live registry.
#[derive(Clone)]
pub struct RunTools {
    registry: ToolRegistry,
    allow: Option<Vec<String>>,
    spec_tools: Option<Vec<String>>,
    /// `Some` only narrows a [`RunKind::Spawned`] view. `None` inherits.
    subagent_tools: Option<Vec<String>>,
    kind: RunKind,
    depth: u32,
    max_subagent_depth: u32,
}

impl RunTools {
    /// Root view: registered names, optional `tools.allow`, optional non-empty
    /// `spec.tools`, then drop `spawn_subagents`.
    ///
    /// An empty `spec.tools` does not cut. An empty `tools.allow` does.
    pub fn resolve(
        registry: ToolRegistry,
        allow: Option<&[String]>,
        spec_tools: Option<&[String]>,
    ) -> Result<Self, Error> {
        Self::resolve_for(
            registry,
            ToolFilter {
                allow,
                spec_tools,
                subagent_tools: None,
                kind: RunKind::Root,
                depth: 0,
                max_subagent_depth: 0,
            },
        )
    }

    /// Allow-list steps, then the kind rule.
    ///
    /// `Root` drops `spawn_subagents`. `Step` keeps it only while
    /// `depth < max_subagent_depth` and the intersection left it in. `Spawned`
    /// intersects `subagent_tools` when that list is set (a name outside the
    /// parent view is an error, not a grant) and drops `spawn_subagents` at
    /// `depth >= max_subagent_depth`.
    pub fn resolve_for(registry: ToolRegistry, filter: ToolFilter<'_>) -> Result<Self, Error> {
        let registered: BTreeSet<String> = registry.names().into_iter().collect();
        let mut visible = apply_allow(&registered, filter.allow)?;
        apply_spec(&mut visible, filter.spec_tools)?;
        if let (RunKind::Spawned, Some(names)) = (filter.kind, filter.subagent_tools) {
            for name in names {
                if !visible.contains(name) {
                    return Err(Error::Config(format!(
                        "spec.subagent_tools includes \"{name}\", which is not available"
                    )));
                }
            }
        }
        // Kind rules run on each lookup. A tool registered after resolve is
        // visible once it passes the same lists.
        Ok(Self {
            registry,
            allow: filter.allow.map(|names| names.to_vec()),
            spec_tools: filter.spec_tools.map(|names| names.to_vec()),
            subagent_tools: filter.subagent_tools.map(|names| names.to_vec()),
            kind: filter.kind,
            depth: filter.depth,
            max_subagent_depth: filter.max_subagent_depth,
        })
    }

    /// Visible tools only, ordered by name.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.registry
            .names()
            .into_iter()
            .filter(|name| self.admits(name))
            .filter_map(|name| self.registry.get(&name).map(|tool| tool.definition()))
            .collect()
    }

    /// The supervisor marked this server down. A re-list removal is not this,
    /// and a name the filter hides stays `not_found`.
    pub fn mcp_down(&self, name: &str) -> bool {
        self.admits(name) && self.registry.is_mcp_down(name)
    }

    /// `None` when `name` fails the filter, even if the process map has it.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        if !self.admits(name) {
            return None;
        }
        self.registry.get(name)
    }

    fn admits(&self, name: &str) -> bool {
        if let Some(allow) = &self.allow {
            if !allow.iter().any(|item| item == name) {
                return false;
            }
        }
        if let Some(spec) = &self.spec_tools {
            if !spec.is_empty() && !spec.iter().any(|item| item == name) {
                return false;
            }
        }
        if let (RunKind::Spawned, Some(subagent)) = (self.kind, &self.subagent_tools) {
            if !subagent.iter().any(|item| item == name) {
                return false;
            }
        }
        if name == SPAWN_SUBAGENTS && self.hides_spawn() {
            return false;
        }
        true
    }

    fn hides_spawn(&self) -> bool {
        match self.kind {
            RunKind::Root => true,
            RunKind::Step | RunKind::Spawned => self.depth >= self.max_subagent_depth,
        }
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
        let tools =
            RunTools::resolve(registry, None, Some(&["zebra".into(), "alpha".into()])).unwrap();
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
        ));
        assert!(err.to_string().contains("missing"), "{err}");

        let err = must_err(RunTools::resolve(
            registry.clone(),
            Some(&["alpha".into()]),
            Some(&["zebra".into()]),
        ));
        assert!(err.to_string().contains("zebra"), "{err}");

        let tools = RunTools::resolve(registry.clone(), Some(&[]), None).unwrap();
        assert!(tools.definitions().is_empty());

        let tools = RunTools::resolve(registry, None, Some(&[])).unwrap();
        assert_eq!(tools.definitions().len(), 2);
    }

    #[test]
    fn root_drops_spawn_subagents_after_the_intersection() {
        let (registry, calls) = registry(&["read_file", "spawn_subagents"]);
        let tools = RunTools::resolve(registry, None, None).unwrap();
        assert!(tools.get("spawn_subagents").is_none());
        assert!(tools.get("read_file").is_some());
        assert_eq!(calls[1].load(Ordering::SeqCst), 0);
    }

    fn names_of(tools: &RunTools) -> Vec<String> {
        tools
            .definitions()
            .into_iter()
            .map(|def| def.name)
            .collect()
    }

    #[test]
    fn step_keeps_spawn_only_under_max_depth_and_an_explicit_list_does_not_grant_it() {
        let (registry, _) = registry(&["read_file", "shell", "spawn_subagents"]);
        let step = RunTools::resolve_for(
            registry.clone(),
            ToolFilter {
                allow: None,
                spec_tools: None,
                subagent_tools: None,
                kind: RunKind::Step,
                depth: 0,
                max_subagent_depth: 2,
            },
        )
        .unwrap();
        assert!(step.get("spawn_subagents").is_some());

        let at_max = RunTools::resolve_for(
            registry.clone(),
            ToolFilter {
                allow: None,
                spec_tools: None,
                subagent_tools: None,
                kind: RunKind::Step,
                depth: 2,
                max_subagent_depth: 2,
            },
        )
        .unwrap();
        assert!(at_max.get("spawn_subagents").is_none());
        assert!(at_max.get("read_file").is_some());

        let cut = RunTools::resolve_for(
            registry,
            ToolFilter {
                allow: None,
                spec_tools: Some(&["read_file".into()]),
                subagent_tools: None,
                kind: RunKind::Step,
                depth: 0,
                max_subagent_depth: 2,
            },
        )
        .unwrap();
        assert_eq!(names_of(&cut), ["read_file"]);
    }

    #[test]
    fn subagent_tools_cannot_grant_and_omitted_inherits_until_max_depth() {
        let (registry, _) = registry(&["read_file", "shell", "spawn_subagents"]);
        let err = match RunTools::resolve_for(
            registry.clone(),
            ToolFilter {
                allow: None,
                spec_tools: Some(&["read_file".into(), "spawn_subagents".into()]),
                subagent_tools: Some(&["shell".into()]),
                kind: RunKind::Spawned,
                depth: 1,
                max_subagent_depth: 2,
            },
        ) {
            Err(err) => err,
            Ok(_) => panic!("expected subagent_tools to refuse a grant"),
        };
        assert!(
            err.to_string().contains("subagent_tools") && err.to_string().contains("shell"),
            "{err}"
        );

        let inherited = RunTools::resolve_for(
            registry.clone(),
            ToolFilter {
                allow: None,
                spec_tools: Some(&["read_file".into(), "spawn_subagents".into()]),
                subagent_tools: None,
                kind: RunKind::Spawned,
                depth: 1,
                max_subagent_depth: 2,
            },
        )
        .unwrap();
        assert_eq!(names_of(&inherited), ["read_file", "spawn_subagents"]);

        let empty = RunTools::resolve_for(
            registry.clone(),
            ToolFilter {
                allow: None,
                spec_tools: Some(&["read_file".into(), "spawn_subagents".into()]),
                subagent_tools: Some(&[]),
                kind: RunKind::Spawned,
                depth: 1,
                max_subagent_depth: 2,
            },
        )
        .unwrap();
        assert!(
            empty.definitions().is_empty(),
            "an empty subagent_tools list cuts the view"
        );

        let at_max = RunTools::resolve_for(
            registry,
            ToolFilter {
                allow: None,
                spec_tools: Some(&["read_file".into(), "spawn_subagents".into()]),
                subagent_tools: Some(&["read_file".into(), "spawn_subagents".into()]),
                kind: RunKind::Spawned,
                depth: 2,
                max_subagent_depth: 2,
            },
        )
        .unwrap();
        assert_eq!(
            names_of(&at_max),
            ["read_file"],
            "spawn_subagents is absent at depth == max even if the list names it"
        );
    }

    #[tokio::test]
    async fn call_refuses_a_name_that_is_registered_but_hidden() {
        let (registry, calls) = registry(&["alpha", "hidden"]);
        let tools = RunTools::resolve(registry, None, Some(&["alpha".into()])).unwrap();
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

    #[test]
    fn definitions_follow_the_live_registry_and_the_kind_rules() {
        let (registry, _) = registry(&["alpha"]);
        let tools = RunTools::resolve(registry.clone(), None, None).unwrap();
        assert_eq!(names_of(&tools), ["alpha"]);

        registry
            .register(Arc::new(Named {
                name: "late",
                calls: Arc::new(AtomicUsize::new(0)),
            }))
            .unwrap();
        registry
            .register(Arc::new(Named {
                name: "spawn_subagents",
                calls: Arc::new(AtomicUsize::new(0)),
            }))
            .unwrap();
        assert_eq!(names_of(&tools), ["alpha", "late"]);
        assert!(tools.get("late").is_some());
        assert!(
            tools.get("spawn_subagents").is_none(),
            "a root still drops spawn_subagents registered after resolve"
        );

        let allowed = RunTools::resolve(registry.clone(), Some(&["alpha".into()]), None).unwrap();
        assert_eq!(names_of(&allowed), ["alpha"]);
        assert!(allowed.get("late").is_none());

        let spec = RunTools::resolve(registry.clone(), None, Some(&["alpha".into()])).unwrap();
        assert_eq!(names_of(&spec), ["alpha"]);

        registry.unregister("alpha");
        assert!(tools.get("alpha").is_none());
        assert_eq!(names_of(&tools), ["late"]);

        registry.mark_mcp_down("late");
        registry.unregister("late");
        assert!(tools.mcp_down("late"));
        assert!(!spec.mcp_down("late"), "a spec-hidden name stays not_found");
        registry.clear_mcp_down(["late"]);
        assert!(!tools.mcp_down("late"));
    }
}
