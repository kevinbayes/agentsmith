use log::debug;
use tonic::Request;
use crate::config::config::Config;
use crate::pb::tei::v1::embed_client::EmbedClient;
use crate::pb::tei::v1::EmbedRequest;
use crate::common::error::WebError;
use crate::common::error::WebResult;

#[derive(Clone)]
pub struct TextEmbeddingGateway {
    base_url: String,
}

impl TextEmbeddingGateway {
    pub fn new(config: Config) -> Self {

        let base_url = config.config.gateways.registry
            .get("text_embedding_gateway")
            .unwrap()
            .baseurl.clone();

        Self { base_url }
    }
}


pub trait GetTextEmbeddings {

    async fn get_text_embeddings(&self, text: &str) -> WebResult<Vec<f32>>;
}

impl GetTextEmbeddings for TextEmbeddingGateway {

    async fn get_text_embeddings(&self, text: &str) -> WebResult<Vec<f32>> {
        let base_url = self.base_url.clone();
        let mut client = EmbedClient::connect(base_url)
            .await
            .map_err(|e| {
                println!("Error connecting to text embedding gateway: {}", e);
                WebError::EmbeddingError { id: 0, code: 0 }
            })
            ?;

        let request = Request::new(EmbedRequest {
           inputs: text.to_string(),
           truncate: false,
           normalize: false,
        });

        let response = client.embed(request)
            .await
            .map_err(|e| {
                println!("Error getting text embeddings: {}", e);
                WebError::EmbeddingError { id: 0, code: 0 }
            })
            ?;

        Ok(response.into_inner().embeddings)
    }
}