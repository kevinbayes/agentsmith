use std::collections::HashMap;
use futures_util::StreamExt;
use qdrant_client::prelude::*;
use qdrant_client::prelude::point_id::PointIdOptions;
use qdrant_client::prelude::point_id::PointIdOptions::Num;
use qdrant_client::serde;
use serde_json::Value as JsonValue;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{CreateCollectionBuilder, VectorParamsBuilder};
use qdrant_client::qdrant::{Value, SearchPoints, SearchResponse, ScoredPoint};

use crate::config::config::Config;
use crate::common::error::{WebError, WebResult};



#[derive(Clone)]
pub struct VectorGateway {
    base_url: String,
}

impl VectorGateway {
    pub fn new(config: Config) -> Self {

        let base_url = config.config.gateways.registry
            .get("vector_gateway")
            .unwrap()
            .baseurl.clone();

        Self { base_url }
    }

    pub async fn init_collections(&self, collections: Vec<String>) -> WebResult<()> {

        let client: Qdrant = Qdrant::from_url(self.base_url.clone().as_str())
            .build()
            .map_err(|e| {
                println!("Error connecting to vector gateway: {}", e);
                WebError::VectorError { id: 0, code: 0 }
            })
            ?;

        for item in collections.iter() {

            let creation_result =
                client
                    .create_collection(
                        CreateCollectionBuilder::new(item.to_string())
                            .vectors_config(VectorParamsBuilder::new(4, Distance::Cosine.into())),
                    )
                    .await;

            match creation_result {
                Ok(_) => println!("Collection {} created", item),
                Err(e) => println!("Error creating collection {}: {}", item, e)
            }
        }

        Ok(())
    }
}


pub trait Index {

    async fn index(&self, collection: String, correlation_id: &u64, embeddings: &Vec<f32>, index_payload: HashMap<String, String>) -> WebResult<()>;
}



impl Index for VectorGateway {

    async fn index(&self, collection: String, correlation_id: &u64, embeddings: &Vec<f32>, index_payload: HashMap<String, String>) -> WebResult<()> {
        
        let base_url = self.base_url.clone();

        let client: QdrantClient = QdrantClient::from_url(base_url.as_str())
            .build()
            .map_err(|e| {
                println!("Error connecting to vector gateway: {}", e);
                WebError::VectorError { id: 0, code: 0 }
            })
            ?;



        let id = correlation_id.clone();
        let sample = embeddings.clone();

        let mut payload: Payload = Payload::new();

        for item in index_payload.iter() {
            payload.insert(item.0.clone(), Value::from(item.1.clone()));
        }

        let points = vec![PointStruct::new(id, sample, payload)];

        let collection_name = collection.clone();

        let response = client.upsert_points_blocking(collection_name, None, points, None)
            .await
            .map_err(|e| {
                println!("Error indexing: {}", e);
                WebError::VectorError { id: 0, code: 0 }
            })
            ?;

        Ok(())
    }
}


pub trait Search {

    async fn search(&self, collection: String, limit: u32, offset: u32, embeddings: Vec<f32>) -> WebResult<Vec<(u64, f32)>>;
}



fn convert_to_vec_response(scored_point: ScoredPoint) -> WebResult<(u64, f32)> {

    match scored_point.id {
        Some(id) => {
            match id.point_id_options {
                Some(Num(options)) => {
                    Ok((options, scored_point.score))
                },
                None => Err(WebError::VectorError { id: 0, code: 0 }),
                _ => Err(WebError::VectorError { id: 0, code: 0 }),
            }
        },
        None => Err(WebError::VectorError { id: 0, code: 0 })
    }
}

impl Search for VectorGateway {

    async fn search(&self, collection: String, limit: u32, offset: u32, embeddings: Vec<f32>) -> WebResult<Vec<(u64, f32)>> {

        let base_url = self.base_url.clone();

        let client: QdrantClient = QdrantClient::from_url(base_url.as_str())
            .build()
            .map_err(|e| {
                println!("Error connecting to vector gateway: {}", e);
                WebError::VectorError { id: 0, code: 0 }
            })
            ?;

        let sample = embeddings.clone();

        let results: SearchResponse = client.search_points(&SearchPoints {
                collection_name: collection,
                vector: sample,
                score_threshold: Some(0.5),
                limit: limit as u64,
                ..Default::default()
            })
            .await
            .map_err(|e| {
                println!("Error searching: {}", e);
                WebError::VectorError { id: 0, code: 0 }
            })
            ?;

        let mut result: Vec<(u64, f32)> = Vec::new();

        for score_point in results.result {

            result.push(convert_to_vec_response(score_point)?);
        }


        Ok(result)
    }
}