use std::collections::BTreeSet;
use std::sync::Arc;

use agentsmith_tools::{Tool, ToolDefinition, ToolRegistry};
use serde_json::Value;

/// A tool about to be registered. `definition().name` must be `namespaced`.
pub struct PreparedTool {
    pub server_name: String,
    pub namespaced: String,
    pub description: String,
    pub parameters: Value,
}

pub struct InstallOutcome {
    pub registered: BTreeSet<String>,
    pub tools_skipped: u32,
}

/// Namespace `{server}__{tool}`, drop names that disappeared, and skip
/// collisions. `previous` is the set this server registered last time, so a
/// name it already owns can be replaced without dropping a foreign tool.
pub fn install_tools(
    registry: &ToolRegistry,
    server: &str,
    defs: Vec<ToolDefinition>,
    previous: &BTreeSet<String>,
    mut make: impl FnMut(PreparedTool) -> Arc<dyn Tool>,
) -> InstallOutcome {
    let mut skipped = 0u32;
    let mut accepted: Vec<(String, Arc<dyn Tool>)> = Vec::new();
    let mut claimed = BTreeSet::new();
    for def in defs {
        if !valid_mcp_tool_name(&def.name) {
            skipped = skipped.saturating_add(1);
            continue;
        }
        let namespaced = format!("{server}__{}", def.name);
        let owned = previous.contains(&namespaced);
        if claimed.contains(&namespaced) || (!owned && registry.get(&namespaced).is_some()) {
            eprintln!("agentsmith: skipping mcp tool {namespaced}: name is already registered");
            skipped = skipped.saturating_add(1);
            continue;
        }
        claimed.insert(namespaced.clone());
        let parameters = object_schema(server, &def.name, &def.parameters);
        let prepared = PreparedTool {
            server_name: def.name,
            namespaced: namespaced.clone(),
            description: def.description,
            parameters,
        };
        accepted.push((namespaced, make(prepared)));
    }
    let next: BTreeSet<String> = accepted.iter().map(|(name, _)| name.clone()).collect();
    for name in previous {
        if !next.contains(name) {
            registry.unregister(name);
        }
    }
    for (name, tool) in accepted {
        if previous.contains(&name) {
            registry.unregister(&name);
        }
        if let Err(err) = registry.register(tool) {
            eprintln!("agentsmith: skipping mcp tool {name}: {err}");
        }
    }
    let registered = next
        .into_iter()
        .filter(|name| registry.get(name).is_some())
        .collect();
    InstallOutcome {
        registered,
        tools_skipped: skipped,
    }
}

fn valid_mcp_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 48
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}

fn object_schema(server: &str, tool: &str, schema: &Value) -> Value {
    if schema.is_object() {
        return schema.clone();
    }
    eprintln!(
        "agentsmith: mcp tool {server}__{tool} schema is not a JSON object; using an open object schema"
    );
    serde_json::json!({ "type": "object", "additionalProperties": true })
}

#[cfg(test)]
mod tests {
    use agentsmith_tools::{ToolCtx, ToolOutput};
    use serde_json::json;

    use super::*;

    struct Fixed {
        name: String,
        body: String,
        parameters: Value,
    }

    #[async_trait::async_trait]
    impl Tool for Fixed {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.clone(),
                description: self.body.clone(),
                parameters: self.parameters.clone(),
            }
        }

        async fn call(
            &self,
            _ctx: ToolCtx,
            _arguments: Value,
        ) -> Result<ToolOutput, agentsmith_core::Error> {
            Ok(ToolOutput {
                content: self.body.clone(),
            })
        }
    }

    fn fixed(name: &str, body: &str) -> Arc<dyn Tool> {
        Arc::new(Fixed {
            name: name.to_string(),
            body: body.to_string(),
            parameters: json!({"type": "object"}),
        })
    }

    fn def(name: &str, parameters: Value) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: name.to_string(),
            parameters,
        }
    }

    #[test]
    fn skips_bad_names_and_a_foreign_collision_and_drops_disappeared_names() {
        let registry = ToolRegistry::new();
        registry.register(fixed("fs__echo", "builtin")).unwrap();
        let first = install_tools(
            &registry,
            "fs",
            vec![
                def("echo", json!({"type": "object"})),
                def("", json!({"type": "object"})),
                def(&"a".repeat(49), json!({"type": "object"})),
                def("has space", json!({"type": "object"})),
                def("read_file", json!(true)),
                def("boom", json!({"type": "object"})),
            ],
            &BTreeSet::new(),
            |prepared| {
                Arc::new(Fixed {
                    name: prepared.namespaced,
                    body: prepared.server_name,
                    parameters: prepared.parameters,
                })
            },
        );
        assert_eq!(first.tools_skipped, 4);
        assert!(registry.get("fs__echo").unwrap().definition().description == "builtin");
        assert!(registry.get("fs__read_file").is_some());
        assert_eq!(
            registry
                .get("fs__read_file")
                .unwrap()
                .definition()
                .parameters,
            json!({ "type": "object", "additionalProperties": true })
        );
        assert!(first.registered.contains("fs__read_file"));
        assert!(first.registered.contains("fs__boom"));
        assert!(!first.registered.contains("fs__echo"));

        let second = install_tools(
            &registry,
            "fs",
            vec![def("boom", json!({"type": "object"}))],
            &first.registered,
            |prepared| fixed(&prepared.namespaced, "replaced"),
        );
        assert!(registry.get("fs__read_file").is_none());
        assert_eq!(
            registry.get("fs__boom").unwrap().definition().description,
            "replaced"
        );
        assert!(registry.get("fs__echo").is_some());
        assert_eq!(second.registered, BTreeSet::from(["fs__boom".into()]));
    }
}
