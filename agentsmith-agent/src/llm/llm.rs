use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prompt {
    pub system: String,
    pub user: String,
}


impl Prompt {
    pub fn new(system: String, user: String) -> Self {
        Self { system, user }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LLMResult {
    pub message: String,
}

impl LLMResult {
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

pub trait GenerateText {

    async fn generate_text(&self, prompt: &Prompt) -> agentsmith_common::error::error::Result<LLMResult>;
}
