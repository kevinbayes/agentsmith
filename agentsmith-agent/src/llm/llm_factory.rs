use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use agentsmith_common::config::config::Config;
use log::info;
use agentsmith_common::error::error::Error::AgentFactoryError;
use crate::llm::anthropic_llm::AnthropicLLM;
use crate::llm::gcp_gemini_llm::GeminiLLM;
use crate::llm::llm::{GenerateText, LLMResult, Prompt};

use crate::llm::cerebras_llm::CerebrasLLM;
use crate::llm::groq_llm::GroqLLM;
use crate::llm::huggingface_tgi_llm::HuggingFaceLLM;
use crate::llm::openai_llm::OpenAILLM;

#[derive(Clone, Debug)]
pub enum LLM {
    AnthropicLLM(AnthropicLLM),
    CerebrasLLM(CerebrasLLM),
    GeminiLLM(GeminiLLM),
    GroqLLM(GroqLLM),
    OpenAILLM(OpenAILLM),
    HuggingFaceLLM(HuggingFaceLLM),
}

trait LLMClient {
    async fn execute(&self, prompt: &Prompt) -> agentsmith_common::error::error::Result<LLMResult>;
}

impl LLMClient for LLM {
    async fn execute(&self, prompt: &Prompt) -> agentsmith_common::error::error::Result<LLMResult> {
        info!("Test");
        match self {
            LLM::AnthropicLLM(llm) => llm.generate(prompt).await,
            LLM::CerebrasLLM(llm) => llm.generate(prompt).await,
            LLM::GeminiLLM(llm) => llm.generate(prompt).await,
            LLM::GroqLLM(llm) => llm.generate(prompt).await,
            LLM::OpenAILLM(llm) => llm.generate(prompt).await,
            LLM::HuggingFaceLLM(llm) => llm.generate(prompt).await,
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
                    "anthropic" => {
                        let llm = &LLM::AnthropicLLM(AnthropicLLM::new(self.config.clone(), String::from("claude-3-5-sonnet-20240620")));
                        registry.register(key.to_string(), llm.clone());

                        Ok(llm.clone())
                    }
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
                    "openai" => {
                        let llm = &LLM::OpenAILLM(OpenAILLM::new(self.config.clone(), String::from("gpt-4o-mini")));
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
    use testcontainers::core::{IntoContainerPort, Mount, WaitFor};
    use testcontainers::{GenericImage, ImageExt};
    use testcontainers::runners::AsyncRunner;
    use agentsmith_common::config::config::read_config;
    use super::*;



    // #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[tokio::test]
    async fn test_call_llm() {

        tracing_subscriber::fmt::init();

        let current_dir = std::env::current_dir().unwrap();

        // let container = GenericImage::new("mockserver/mockserver", "latest")
        //     .with_exposed_port(1080.tcp())
        //     .with_wait_for(WaitFor::message_on_stdout("Ready to accept connections"))
        //     .with_mount(Mount::bind_mount(format!("{}/resources/mockserver/", current_dir.to_str().expect("Need a directory.")), "/config"))
        //     .start()
        //     .await
        //     .expect("Mockserver started.");

        let config = read_config("./secret-config.json").unwrap();

        let factory = LLMFactory::new(config);

        let result5 = factory.instance("openai").unwrap().execute(&Prompt {
            user: String::from("Tell me what your job is?"),
            system: String::from("You are a helpful assistant to a financial banker who screens fraudulent individuals.")
        }).await.unwrap();
        info!("anthropic: {:?}", result5);

        // let result4 = factory.instance("anthropic").unwrap().execute(&Prompt {
        //     user: String::from("Tell me what your job is?"),
        //     system: String::from("You are a helpful assistant to a financial banker who screens fraudulent individuals.")
        // }).await.unwrap();
        // info!("anthropic: {:?}", result4);

        // let result = factory.instance("cerebras").unwrap().execute(&Prompt {
        //     user: String::from("Tell me what your job is?"),
        //     system: String::from("You are a helpful assistant to a financial banker who screens fraudulent individuals.")
        // }).await.unwrap();
        // info!("cerebras: {:?}", result);
        //
        // let result2 = factory.instance("groq").unwrap().execute(&Prompt {
        //     user: String::from("Tell me what your job is?"),
        //     system: String::from("You are a helpful assistant to a financial banker who screens fraudulent individuals.")
        // }).await.unwrap();
        // info!("groq: {:?}", result2);

        // let result3 = factory.instance("gemini").unwrap().execute(&Prompt {
        //     user: String::from("Tell me what your job is?"),
        //     system: String::from("You are a helpful assistant to a financial banker who screens fraudulent individuals.")
        // }).await.unwrap();




        // info!("gemini: {:?}", result3);
        // assert_eq!(result.message.clone(), result.message);
    }
}