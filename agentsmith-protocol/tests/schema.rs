use serde_json::{json, Value};

fn validator(schema: &Value) -> jsonschema::Validator {
    jsonschema::validator_for(schema).expect("schema compiles")
}

fn errors(validator: &jsonschema::Validator, value: &Value) -> Vec<String> {
    validator
        .iter_errors(value)
        .map(|err| format!("{}: {err}", err.instance_path))
        .collect()
}

fn assert_valid(validator: &jsonschema::Validator, value: &Value) {
    let found = errors(validator, value);
    assert!(found.is_empty(), "{found:?}\n{value}");
}

fn assert_invalid(validator: &jsonschema::Validator, value: &Value) {
    assert!(
        validator.validate(value).is_err(),
        "expected schema rejection\n{value}"
    );
}

#[test]
fn design_examples_match_the_closed_schema() {
    let schema: Value =
        serde_json::from_str(include_str!("../schema/ahsp-1.0.json")).expect("schema json");
    let validator = validator(&schema);
    let idle: Value =
        serde_json::from_str(include_str!("fixtures/idle.json")).expect("idle example");
    let plan: Value =
        serde_json::from_str(include_str!("fixtures/plan.json")).expect("plan example");
    assert_valid(&validator, &idle);
    assert_valid(&validator, &plan);

    let mut builtin = idle.clone();
    builtin["tools"] = json!([{ "name": "shell", "source": "builtin", "enabled": true }]);
    assert_valid(&validator, &builtin);

    let mut builtin_server = builtin.clone();
    builtin_server["tools"][0]["server"] = json!("fs");
    assert_invalid(&validator, &builtin_server);

    let mut mcp_missing = idle.clone();
    mcp_missing["tools"] = json!([{ "name": "fs__read_file", "source": "mcp", "enabled": true }]);
    assert_invalid(&validator, &mcp_missing);

    let mut mcp_ok = mcp_missing.clone();
    mcp_ok["tools"][0]["server"] = json!("fs");
    assert_valid(&validator, &mcp_ok);
}
