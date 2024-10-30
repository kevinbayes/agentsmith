use std::sync::Arc;
use axum::Json;
use r2d2_redis::RedisConnectionManager;
use sqlx::{MySql, MySqlPool, Pool};
use tokio::sync::RwLock;
use agentsmith_common::redis::common::RedisPool;
use crate::common::error::{WebError, WebResult};
use crate::model::account::Account;
use agentsmith_common::model::common::{Page, PagedResult};
use crate::model::party::PartyCreated;
use crate::model::person::{Person};
use crate::service::account::get_account::get_account_by_reference;
use crate::service::people::get_person::get_person_by_id;

#[derive(Clone)]
pub struct AccountService {
    redis_pool: Arc<RedisPool>,
    database_pool: MySqlPool,
}

impl AccountService {
    pub fn new(redis_pool: RedisPool, database_pool: MySqlPool) -> AccountService {
        AccountService { redis_pool: Arc::new(redis_pool), database_pool: database_pool }
    }
}

impl AccountService {

    pub async fn create(&self, ) -> WebResult<Json<PartyCreated>> {
        Ok(Json(PartyCreated { id: 0 }))
    }

    pub async fn list(&self, ) -> WebResult<Json<PagedResult<Account>>> {
        Ok(Json(PagedResult { page: Page {page: 0, size: 0, total:0 }, items: vec![] }))
    }

    pub async fn get(&self, reference: String, tenant: String) -> WebResult<Json<Account>> {

        match get_account_by_reference("".to_string(), reference, tenant, &self.database_pool, &self.redis_pool).await {
            Some(account) => Ok(Json(account)),
            _ => Err(WebError::AccountError {id: 0, code: 0})
        }
    }
}