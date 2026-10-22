use serde::Deserialize;
use serde_json::{json, Value};

use agentsmith_core::Error;

use super::{cap_output, clamp_output_bytes, DEFAULT_MAX_OUTPUT_BYTES};
use crate::tool::{ensure_not_cancelled, Tool, ToolCtx, ToolDefinition, ToolOutput};

const NAME: &str = "finish";

/// Return the model's answer. Does not touch the filesystem.
#[derive(Debug)]
pub struct Finish {
    max_output_bytes: u64,
}

impl Default for Finish {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_OUTPUT_BYTES)
    }
}

impl Finish {
    pub fn new(max_output_bytes: u64) -> Self {
        Self {
            max_output_bytes: clamp_output_bytes(max_output_bytes),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    answer: String,
}

#[async_trait::async_trait]
impl Tool for Finish {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: NAME.to_string(),
            description: "End this loop successfully with the final answer.".to_string(),
            parameters: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["answer"],
                "properties": {
                    "answer": { "type": "string" }
                }
            }),
        }
    }

    async fn call(&self, ctx: ToolCtx, arguments: Value) -> Result<ToolOutput, Error> {
        ensure_not_cancelled(NAME, &ctx)?;
        let args: Args = super::parse_args(NAME, arguments)?;
        Ok(ToolOutput {
            content: cap_output(&args.answer, self.max_output_bytes, false),
        })
    }
}
