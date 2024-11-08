use std::collections::HashMap;
use std::iter::Map;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Response;
use serde_json::{json, Value};
use short_uuid::ShortUuid;
use agentsmith_common::error::error::SystemResult;
use crate::agent::agent::AgentConfig;
use crate::agent::agent_tool::AgentTool;
use crate::llm::llm_factory::LLM;
use crate::memory::memory::Memory;
use crate::tools::tool::{SimpleToolExecution, ToolResult, ToolType};

#[derive(Clone)]
pub struct SimpleJsonWebClientTool {
    pub r#type: ToolType,
    pub code: String,
    pub description: String,
    pub url: String,
    pub method: String,
    pub input_schema: Value,
    pub output_schema: Value,
    pub post_process: for<'a> fn(&SimpleJsonWebClientTool, &'a Value, &'a Value) -> &'a Value,
    client: Arc<reqwest::Client>
}


impl SimpleJsonWebClientTool {
    
    fn new(
        code: String,
        description: String,
        url: String,
        method: String,
        headers: HashMap<String, String>,
        connect_timeout: u64,
        read_timeout: u64,
        timeout: u64,
        input_schema: Value,
        output_schema: Value,
        post_process: for<'a> fn(&SimpleJsonWebClientTool, &'a Value, &'a Value) -> &'a Value,
    ) -> Self {

        let client = reqwest::ClientBuilder::new()
            .timeout(Duration::from_secs(timeout))
            .connect_timeout(Duration::from_secs(connect_timeout))
            .read_timeout(Duration::from_secs(read_timeout))
            .default_headers(SimpleJsonWebClientTool::create_headers(headers))
            .build()
            .unwrap();

        Self {
            r#type: ToolType::Function,
            code,
            description,
            url,
            method,
            input_schema,
            output_schema,
            post_process,
            client: Arc::new(client)
        }
    }

    fn create_headers(headers: HashMap<String, String>) -> HeaderMap {
        let mut header_map = HeaderMap::new();

        for (key, value) in headers {
            // Convert string to HeaderName and HeaderValue, handling potential parse errors
            if let (Ok(name), Ok(val)) = (
                HeaderName::from_bytes(key.as_bytes()),
                HeaderValue::from_str(&value)
            ) {
                header_map.insert(name, val);
            }
        }

        header_map
    }

    async fn handle_post(&self, input: &Value) -> reqwest::Result<Response> {

        let request = self.client.post(self.url.clone())
            .json(input)
            .build()?;

        self.client.execute(request).await
    }

    async fn handle_put(&self, input: &Value) -> reqwest::Result<Response> {

        let request = self.client.put(self.url.clone())
            .json(input)
            .build()?;

        self.client.execute(request).await
    }

    async fn handle_get(&self, input: &Value) -> reqwest::Result<Response> {

        let request = self.client.get(self.url.clone())
            .json(input)
            .build()?;

        self.client.execute(request).await
    }

    async fn handle_delete(&self, input: &Value) -> reqwest::Result<Response> {

        let request = self.client.delete(self.url.clone())
            .json(input)
            .build()?;

        self.client.execute(request).await
    }

    async fn response_body(&self, response: Response) -> Value {

        let headers = response.headers().clone();

        if let Some(content_length) = response.content_length() {
            if content_length == 0 {
               println!("Content is empty!");
               return json!({});
            }
        }

        let  content_type = headers.get(reqwest::header::CONTENT_TYPE)
            .map(|h| h.to_str().unwrap_or("application/json"))
            .unwrap_or("application/json");


        match content_type {
            "application/json" => {
                response.json().await.unwrap_or(json!({}))
            },
            _ => json!({"text": response.text().await.unwrap_or("".to_string())})
        }
    }
}

impl SimpleToolExecution for SimpleJsonWebClientTool {


    async fn execute(&self, id: Option<String>, input: &Value) -> SystemResult<ToolResult> {

        let method= self.method.clone();

        let result = match method.as_str() {
            "option" => self.handle_post(input).await,
            "post" => self.handle_post(input).await,
            "put" => self.handle_put(input).await,
            "delete" => self.handle_delete(input).await,
            _ => self.handle_get(input).await
        };

        let final_value = match result {
            Ok(response) => {
                let status_code = response.status().as_u16();
                let body = self.response_body(response).await;
                json!({ "status_code": status_code, "body": body })
            }
            Err(error) => {
                json!({ "status_code": 500, "error": format!("Error calling api due to: {:?}", error) })
            }
        };

        Ok(ToolResult {
            id: id.unwrap_or(ShortUuid::generate().to_string()),
            code: self.code.clone(),
            value: final_value.clone(),
        })
    }
}