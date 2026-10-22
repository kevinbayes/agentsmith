use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

// After the prefix, consume `_`, `-`, and base64 `+/=` until whitespace.
// `.` is excluded so a sentence period stays, except Bearer, whose alphabet
// includes `.` and `~` (and `=`, so `Bearer ab=cd` does not leave `cd`).
// `sk-` does not demand 8 alphanumerics before the first `-` (`sk-proj-`).
static SECRET_TEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"sk-[A-Za-z0-9_+/=-]{8,}",
        r"|github_pat_[A-Za-z0-9_+/=-]*",
        r"|ghp_[A-Za-z0-9_+/=-]*",
        r"|gho_[A-Za-z0-9_+/=-]*",
        r"|xox[baprs]-[A-Za-z0-9_+/=-]*",
        r"|xapp-[A-Za-z0-9_+/=-]*",
        r"|Bearer [A-Za-z0-9_+/=.~-]+",
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
        let tailed = json!({"note": "see ghp_abcDEF1234567890_TAIL now"});
        assert_eq!(redact_value(&tailed)["note"], "see [redacted] now");
    }

    #[test]
    fn redact_consumes_the_rest_of_the_token() {
        for token in [
            "sk-testsecret-MORESECRET",
            "ghp_abcDEF1234567890_TAIL",
            "ghp_abc_def_TAIL",
            "sk-ant-api03-secret+plus/slash=",
            "github_pat_11AAAA_secret-TAIL",
            "xoxb-1234-abcdef_TAIL",
            "Bearer ab=cd",
            "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
        ] {
            assert_eq!(redact(token), "[redacted]", "{token}");
        }
        assert_eq!(redact("ghp_abcDEF1234567890."), "[redacted].");
        assert_eq!(
            redact("prefix sk-testsecret-MORESECRET."),
            "prefix [redacted]."
        );
    }
}
