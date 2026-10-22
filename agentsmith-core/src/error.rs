use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("config: {0}")]
    Config(String),
    #[error("llm {kind}: {message}")]
    Llm { kind: LlmErrorKind, message: String },
    #[error("tool {tool} {kind}: {message}")]
    Tool {
        tool: String,
        kind: ToolErrorKind,
        message: String,
    },
    #[error("mcp server {server}: {message}")]
    Mcp { server: String, message: String },
    #[error("loop: {0}")]
    Loop(String),
    #[error("cancelled")]
    Cancelled,
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlmErrorKind {
    Auth,
    BadRequest,
    Unavailable,
    Timeout,
}

impl std::fmt::Display for LlmErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Auth => "auth",
            Self::BadRequest => "bad_request",
            Self::Unavailable => "unavailable",
            Self::Timeout => "timeout",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolErrorKind {
    NotFound,
    InvalidArgs,
    Execution,
    Timeout,
    Cancelled,
    McpDown,
    Limit,
}

impl std::fmt::Display for ToolErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NotFound => "not_found",
            Self::InvalidArgs => "invalid_args",
            Self::Execution => "execution",
            Self::Timeout => "timeout",
            Self::Cancelled => "cancelled",
            Self::McpDown => "mcp_down",
            Self::Limit => "limit",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_matches_the_public_strings() {
        assert_eq!(
            Error::Llm {
                kind: LlmErrorKind::Auth,
                message: "no".into(),
            }
            .to_string(),
            "llm auth: no"
        );
        assert_eq!(
            Error::Tool {
                tool: "shell".into(),
                kind: ToolErrorKind::Timeout,
                message: "slow".into(),
            }
            .to_string(),
            "tool shell timeout: slow"
        );
        assert_eq!(
            Error::Mcp {
                server: "fs".into(),
                message: "down".into(),
            }
            .to_string(),
            "mcp server fs: down"
        );
        assert_eq!(Error::Cancelled.to_string(), "cancelled");
        assert_eq!(Error::Config("bad".into()).to_string(), "config: bad");
        assert_eq!(Error::Loop("x".into()).to_string(), "loop: x");
    }

    #[test]
    fn tool_error_kind_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&ToolErrorKind::NotFound).unwrap(),
            "\"not_found\""
        );
        assert_eq!(
            serde_json::to_string(&ToolErrorKind::InvalidArgs).unwrap(),
            "\"invalid_args\""
        );
        assert_eq!(
            serde_json::to_string(&ToolErrorKind::McpDown).unwrap(),
            "\"mcp_down\""
        );
        assert_eq!(
            serde_json::to_string(&ToolErrorKind::Limit).unwrap(),
            "\"limit\""
        );
    }
}
