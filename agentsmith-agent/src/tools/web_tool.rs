use crate::tools::tool::{SimpleToolExecution, ToolResult, ToolType};
use agentsmith_common::error::error::SystemResult;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Response;
use serde_json::{json, Value};
use short_uuid::ShortUuid;
use std::collections::HashMap;
use std::ops::Index;
use std::sync::Arc;
use std::time::Duration;

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
    client: Arc<reqwest::Client>,
}

fn default_post_processor<'a>(this: &SimpleJsonWebClientTool, input: &'a Value, result: &'a Value) -> &'a Value {
    result
}

impl SimpleJsonWebClientTool {
    pub fn new(
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
        post_process: Option<for<'a> fn(this: &SimpleJsonWebClientTool, input: &'a Value, result: &'a Value) -> &'a Value>,
    ) -> Self {
        let client = reqwest::ClientBuilder::new()
            .timeout(Duration::from_secs(timeout))
            .connect_timeout(Duration::from_secs(connect_timeout))
            .read_timeout(Duration::from_secs(read_timeout))
            .default_headers(SimpleJsonWebClientTool::create_headers(headers))
            .build()
            .unwrap();

        let post_process = if let Some(post_process_function) = post_process {
            post_process_function
        } else {
            default_post_processor
        };

        Self {
            r#type: ToolType::Function,
            code,
            description,
            url,
            method,
            input_schema,
            output_schema,
            post_process,
            client: Arc::new(client),
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

        let content_type = headers.get(reqwest::header::CONTENT_TYPE)
            .map(|h| h.to_str().unwrap_or("application/json"))
            .unwrap_or("application/json");

        println!("content type: {}", content_type);

        match content_type.to_lowercase().contains("application/json") {
            true => {
                response.json().await.unwrap_or(json!({}))
            }
            _ => json!({"text": response.text().await.unwrap_or("".to_string())})
        }
    }
}

impl SimpleToolExecution for SimpleJsonWebClientTool {
    async fn execute(&self, id: Option<String>, input: &Value) -> SystemResult<ToolResult> {
        let method = self.method.clone();

        let result = match method.as_str() {
            "post" => self.handle_post(input).await,
            "put" => self.handle_put(input).await,
            "delete" => self.handle_delete(input).await,
            _ => self.handle_get(input).await
        };

        let final_value = match result {
            Ok(response) => {
                let status_code = response.status().as_u16();
                let body = self.response_body(response).await;
                let body = (self.post_process)(self, input, &body);
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

    fn format_result(&self, result_value: Value) -> Vec<String> {
        println!("Tool result recording: {:?}", result_value);
        if result_value.is_null() {
            vec![]
        } else if result_value.is_object() {
            let mut messages: Vec<String> = vec![];

            if let Some(result) = result_value.get("body") {
                messages.push(result.clone().to_string());
            }

            if let Some(error) = result_value.get("error") {
                messages.push(error.as_str().unwrap().to_string());
            }

            messages
        } else {

            vec![result_value.to_string()]
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};


    // #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[tokio::test]
    async fn test_post_request() {

        //Given
        let code = "post-weather".to_string();


        let tool = SimpleJsonWebClientTool::new(
            code.clone(),
            "Get the weather for the unit test".to_string(),
            "http://localhost:1080/agentsmith-agent/unittest/tools/web-tool/1".to_string(),
            "post".to_string(),
            HashMap::new(),
            1000,
            1000,
            1000,
            Value::Null,
            json!({
          "type": "object",
          "properties": {
            "message": {
              "type": "string",
              "description": "General message"
            },
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "temp": {
              "type": "object",
              "properties": {
                    "unit": {
                                "type": "string",
                                "description": "Unit of measure"
                            },
                    "amount": {
                                "type": "number",
                                "description": "numeric representation"
                            }
              }
            }
          },
          "required": ["message"]
        }),
            |tool, input, response| { todo!() },
        );

        let input = json!({
            "location": "South Africa, Johannesburg",
            "unit": "celsius"
        });

        let result = tool.execute(Some("1".to_string()), &input).await.unwrap();

        println!("{:?}", result);

        assert_eq!("1".to_string(), result.id);

        assert_eq!(code.clone(), result.code);

        let expected_status_code: i64 = 200;
        let actual_status_code: i64 = result.value.get("status_code").unwrap().as_i64().unwrap();
        assert_eq!(expected_status_code, actual_status_code, "Expected 200 status code.");

        let expected_body = json!({
            "message": "hello world",
            "location": "Test",
            "temp": {
              "unit": "celsius",
              "amount": 100.0
            }
          });
        let actual_body: Value = result.value.get("body").unwrap().clone();
        assert_eq!(expected_body, actual_body, "Expected response did not match");
    }

    // #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[tokio::test]
    async fn test_get_request() {

        //Given
        let code = "get-weather".to_string();


        let tool = SimpleJsonWebClientTool::new(
            code.clone(),
            "Get the weather for the unit test".to_string(),
            "http://localhost:1080/agentsmith-agent/unittest/tools/web-tool/1".to_string(),
            "get".to_string(),
            HashMap::new(),
            1000,
            1000,
            1000,
            Value::Null,
            json!({
          "type": "object",
          "properties": {
            "message": {
              "type": "string",
              "description": "General message"
            },
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "temp": {
              "type": "object",
              "properties": {
                    "unit": {
                                "type": "string",
                                "description": "Unit of measure"
                            },
                    "amount": {
                                "type": "number",
                                "description": "numeric representation"
                            }
              }
            }
          },
          "required": ["message"]
        }),
            |tool, input, response| { todo!() },
        );

        let input = json!({
            "location": "South Africa, Johannesburg",
            "unit": "celsius"
        });

        let result = tool.execute(Some("1".to_string()), &input).await.unwrap();

        println!("{:?}", result);

        assert_eq!("1".to_string(), result.id);

        assert_eq!(code.clone(), result.code);

        let expected_status_code: i64 = 200;
        let actual_status_code: i64 = result.value.get("status_code").unwrap().as_i64().unwrap();
        assert_eq!(expected_status_code, actual_status_code, "Expected 200 status code.");

        let expected_body = json!({
            "message": "hello world",
            "location": "Test",
            "temp": {
              "unit": "celsius",
              "amount": 100.0
            }
          });
        let actual_body: Value = result.value.get("body").unwrap().clone();
        assert_eq!(expected_body, actual_body, "Expected response did not match");
    }
}