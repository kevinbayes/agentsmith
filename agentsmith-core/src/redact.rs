use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

// Consume the token body after each prefix. Replacing only `ghp_` would leave the secret.
static SECRET_TEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"sk-ant-[A-Za-z0-9_-]*",
        r"|sk-[A-Za-z0-9]{8,}",
        r"|github_pat_[A-Za-z0-9_]*",
        r"|ghp_[A-Za-z0-9]*",
        r"|gho_[A-Za-z0-9]*",
        r"|xox[baprs]-[A-Za-z0-9-]*",
        r"|xapp-[A-Za-z0-9-]*",
        r"|Bearer [A-Za-z0-9\-._~+/]+=*",
        r"|AKIA[0-9A-Z]{16}",
    ))
    .expect("secret text pattern")
});

static SECRET_KEY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(api[_-]?key|token|secret|password|passwd|authorization|credential|private[_-]?key)",
    )
    .expect("secret key pattern")
});

pub fn redact(text: &str) -> String {
    SECRET_TEXT.replace_all(text, "[redacted]").into_owned()
}

pub fn redact_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (key, child) in map {
                if SECRET_KEY.is_match(key) {
                    out.insert(key.clone(), Value::String("[redacted]".to_string()));
                } else {
                    out.insert(key.clone(), redact_value(child));
                }
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(redact_value).collect()),
        Value::String(text) => Value::String(redact(text)),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn redact_removes_planted_sk_and_ghp_tokens() {
        assert_eq!(
            redact("prefix sk-testsecret suffix"),
            "prefix [redacted] suffix"
        );
        assert_eq!(redact("token ghp_abcDEF1234567890."), "token [redacted].");
        assert!(!redact("sk-ant-api03-secretvalue").contains("secretvalue"));
        assert_eq!(redact("hello world"), "hello world");
    }

    #[test]
    fn redact_covers_the_other_token_shapes() {
        assert_eq!(
            redact("Authorization: Bearer abc.def-ghi"),
            "Authorization: [redacted]"
        );
        assert_eq!(redact("key AKIAIOSFODNN7EXAMPLE end"), "key [redacted] end");
        assert_eq!(redact("xoxb-1234-abcdef"), "[redacted]");
        assert_eq!(redact("xoxs-1234-abcdef"), "[redacted]");
        assert!(!redact("github_pat_11AAAAAAA0abc").contains("11AAAAAAA0abc"));
        assert_eq!(redact("gho_abcDEF123"), "[redacted]");
        assert_eq!(redact("xapp-1-abc"), "[redacted]");
    }

    #[test]
    fn redact_value_replaces_sensitive_keys_and_string_secrets() {
        let value = json!({
            "api_key": "super-secret",
            "nested": { "client_secret": { "x": 1 } },
            "token_url": "https://auth.example/token",
            "note": "prefix sk-testsecret suffix",
            "items": ["ghp_abcDEF1234567890"]
        });
        let out = redact_value(&value);
        assert_eq!(out["api_key"], "[redacted]");
        assert_eq!(out["nested"]["client_secret"], "[redacted]");
        assert_eq!(out["token_url"], "[redacted]");
        assert_eq!(out["note"], "prefix [redacted] suffix");
        assert_eq!(out["items"][0], "[redacted]");
    }
}
