use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use agentsmith_common::config::config::Config;
use log::info;
use agentsmith_common::error::error::Error::AgentFactoryError;
use crate::llm::gcp_gemini_llm::GeminiLLM;
use crate::llm::llm::{GenerateText, LLMResult, Prompt};

use crate::llm::cerebras_llm::CerebrasLLM;
use crate::llm::groq_llm::GroqLLM;
use crate::llm::huggingface_tgi_llm::HuggingFaceLLM;

#[derive(Clone, Debug)]
pub enum LLM {
    CerebrasLLM(CerebrasLLM),
    GeminiLLM(GeminiLLM),
    GroqLLM(GroqLLM),
    HuggingFaceLLM(HuggingFaceLLM),
}

trait LLMClient {
    async fn execute(&self, prompt: &Prompt) -> agentsmith_common::error::error::Result<LLMResult>;
}

impl LLMClient for LLM {
    async fn execute(&self, prompt: &Prompt) -> agentsmith_common::error::error::Result<LLMResult> {
        info!("Test");
        match self {
            LLM::CerebrasLLM(llm) => llm.generate_text(prompt).await,
            LLM::GeminiLLM(llm) => llm.generate_text(prompt).await,
            LLM::GroqLLM(llm) => llm.generate_text(prompt).await,
            LLM::HuggingFaceLLM(llm) => llm.generate_text(prompt).await,
        }
    }
}

pub struct LLMFactory {
    config: Config,
    registry: Arc<Mutex<LLMRegistry>>
}

#[derive(Clone, Debug)]
pub struct Agent {
    llm: Arc<LLM>
}

struct LLMRegistry {
    items: HashMap<String, LLM>,
}

impl LLMRegistry {
    fn new() -> Self {
        LLMRegistry { items: HashMap::new() }
    }

    fn register(&mut self, key: String, item: LLM) {
        self.items.insert(key, item);
    }

    fn get(&self, key: &str) -> Option<&LLM> {
        self.items.get(key)
    }
}

impl LLMFactory {

    pub fn new(config: Config) ->  Self {



        Self {
            config: config.clone(),
            registry: Arc::new(Mutex::new(LLMRegistry::new())),
        }
    }

    pub fn instance(&self, key: &str) -> agentsmith_common::error::error::Result<LLM> {

        let mut registry = self.registry.lock().unwrap();

        match registry.get(key) {
            Some(llm) => {
                Ok(llm.clone())
            }
            _ => {
                match key {
                    "cerebras" => {
                        let llm = &LLM::CerebrasLLM(CerebrasLLM::new(self.config.clone(), String::from("llama3.1-8b")));
                        registry.register(key.to_string(), llm.clone());

                        Ok(llm.clone())
                    }
                    "gemini" => {
                        let llm = &LLM::GeminiLLM(GeminiLLM::new(self.config.clone()));
                        registry.register(key.to_string(), llm.clone());
                        Ok(llm.clone())
                    }
                    "groq" => {
                        let llm = &LLM::GroqLLM(GroqLLM::new(self.config.clone(), String::from("llama-3.1-8b-instant")));
                        registry.register(key.to_string(), llm.clone());
                        Ok(llm.clone())
                    }
                    _ => {
                        Err(AgentFactoryError {id: 0, code: 0})
                    }
                }
            }
        }
    }
}




#[cfg(test)]
mod tests {
    use std::fs;
    use agentsmith_common::config::config::read_config;
    use super::*;



    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_call_llm() {

        tracing_subscriber::fmt::init();

        let config = read_config("./secret-config.json").unwrap();

        let factory = crate::llm::llm_factory::LLMFactory::new(config);

        let result = factory.instance("cerebras").unwrap().execute(&Prompt {
            user: String::from("Tell me what your job is?"),
            system: String::from("You are a helpful assistant to a financial banker who screens fraudulent individuals.")
        }).await.unwrap();

        let result2 = factory.instance("groq").unwrap().execute(&Prompt {
            user: String::from("Tell me what your job is?"),
            system: String::from("You are a helpful assistant to a financial banker who screens fraudulent individuals.")
        }).await.unwrap();

        let result3 = factory.instance("gemini").unwrap().execute(&Prompt {
            user: String::from("Tell me what your job is?"),
            system: String::from("You are a helpful assistant to a financial banker who screens fraudulent individuals.")
        }).await.unwrap();

        info!("cerebras: {:?}", result);
        info!("groq: {:?}", result2);
        info!("gemini: {:?}", result3);
        assert_eq!(result.message.clone(), result.message);
    }
}