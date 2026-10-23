use serde_json::{json, Value};

use crate::types::ToolCall;

pub(crate) fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// A string is parsed. Invalid JSON keeps the original text and `arguments: None`.
/// An object (or any other JSON value) is stored as compact text and `Some`.
pub(crate) fn split_arguments(value: Option<&Value>) -> (String, Option<Value>) {
    match value {
        None => (String::new(), None),
        Some(Value::String(text)) => {
            let parsed = serde_json::from_str(text).ok();
            (text.clone(), parsed)
        }
        Some(other) => (compact_json(other), Some(other.clone())),
    }
}

/// Anthropic `input` and Gemini `args` must be objects. Anything else becomes `{}`.
pub(crate) fn object_arguments(call: &ToolCall) -> Value {
    if let Some(Value::Object(map)) = &call.arguments {
        return Value::Object(map.clone());
    }
    match serde_json::from_str::<Value>(&call.arguments_raw) {
        Ok(Value::Object(map)) => Value::Object(map),
        _ => json!({}),
    }
}
