use std::collections::HashMap;
use agentsmith_common::config::config::Config;
use crate::llm::cerebras_llm::CerebrasLLM;
use crate::llm::gcp_gemini_llm::GeminiLLM;
use crate::llm::groq_llm::GroqLLM;
use crate::llm::huggingface_tgi_llm::HuggingFaceLLM;
use crate::llm::llm_factory::{LLMFactory, LLM};

struct AgentFactory {
    config: Config
}

impl AgentFactory {

    pub fn new(config: Config) ->  Self {
        Self {
            config: config.clone(),
        }
    }
}