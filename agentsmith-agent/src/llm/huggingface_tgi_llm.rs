use crate::llm::llm::{GenerateText, LLMConfiguration, LLMResult};
use crate::llm::prompt::Prompt;
use agentsmith_common::config::config::Config;
use agentsmith_common::error::error::Result;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct HuggingFaceLLM {
    api_key: String,
    base_url: String,
    client: Arc<reqwest::Client>,
}

impl HuggingFaceLLM {

    pub fn new(config: Config, llm_configuration: LLMConfiguration) -> Self {

        let gemini_config = config.config.gateways.registry
            .get("gemini_gateway")
            .unwrap();

        let api_key = gemini_config.api_key.clone();
        let base_url = gemini_config.baseurl.clone();
        let client = Arc::new(reqwest::ClientBuilder::new()
            .connect_timeout(Duration::from_secs(60))
            .build()
            .unwrap());

        Self { base_url, api_key, client }
    }


}

impl GenerateText for HuggingFaceLLM {

    async fn generate(&self, prompt: &Prompt) -> Result<LLMResult> {

        Ok(LLMResult::new(String::from("Static!")))
    }
}