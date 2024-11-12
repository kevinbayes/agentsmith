use fastembed::TextEmbedding;
use sqlx::mysql::MySqlPoolOptions;
use sqlx::MySqlPool;
use crate::config::config::{Config, DatabaseConfig};
use crate::gateway::text_embedding_gateway::TextEmbeddingGateway;
use crate::gateway::vector_gateway::VectorGateway;
use crate::model::configuration::Configuration;

#[derive(Clone)]
pub struct GatewayRegistry {
    pub text_embedding_gateway: TextEmbeddingGateway,
    pub vector_gateway: VectorGateway,
}

impl GatewayRegistry {
    pub async fn new(config: Config) -> Self {

        let text_embedding_gateway = TextEmbeddingGateway::new(config.clone());
        let vector_gateway = VectorGateway::new(config.clone());

        let _ = vector_gateway.init_collections(vec![]).await;

        Self { text_embedding_gateway, vector_gateway }
    }
}


pub async fn create_gateway_registry(config: Config) -> GatewayRegistry {

    GatewayRegistry::new(config).await
}