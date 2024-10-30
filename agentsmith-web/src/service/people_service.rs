use std::sync::Arc;
use axum::Json;
use r2d2_redis::RedisConnectionManager;
use sqlx::{MySql, MySqlPool, Pool};
use tokio::sync::RwLock;
use agentsmith_common::redis::common::RedisPool;
use crate::common::error::{WebError, WebResult};
use agentsmith_common::model::common::{Page, PagedResult};
use crate::model::party::PartyCreated;
use crate::model::person::{Person};
use crate::service::people::get_person::get_person_by_id;

#[derive(Clone)]
pub struct PersonService {
    redis_pool: Arc<RedisPool>,
    database_pool: MySqlPool,
}

impl PersonService {
    pub fn new(redis_pool: RedisPool, database_pool: MySqlPool) -> PersonService {
        PersonService { redis_pool: Arc::new(redis_pool), database_pool: database_pool }
    }
}

impl PersonService {

    pub async fn create(&self, ) -> WebResult<Json<PartyCreated>> {
        Ok(Json(PartyCreated { id: 0 }))
    }

    pub async fn list(&self, ) -> WebResult<Json<PagedResult<Person>>> {
        Ok(Json(PagedResult { page: Page {page: 0, size: 0, total:0 }, items: vec![] }))
    }

    pub async fn get(&self, id: String, tenant: String) -> WebResult<Json<Person>> {

        match get_person_by_id(id, tenant, &self.database_pool, &self.redis_pool).await {
            Some(person) => Ok(Json(person)),
            _ => Err(WebError::PersonError {id: 0,  code: 0})
        }
    }
}