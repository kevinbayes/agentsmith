use serde_json::Value;
use agentsmith_common::error::error::SystemResult;

pub trait SimpleToolExecution {

    async fn execute(&self, input: &Value) -> SystemResult<Value>;
}