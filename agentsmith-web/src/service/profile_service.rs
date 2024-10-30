use std::sync::Arc;
use axum::Json;
use chrono::{DateTime, Utc};
use log::warn;
use r2d2_redis::RedisConnectionManager;
use sqlx::{MySql, MySqlPool, Pool};
use tokio::sync::RwLock;
use agentsmith_common::redis::common::RedisPool;
use crate::common::error::{WebError, WebResult};
use crate::model::account::{Account, CreateProfile, Profile, ProfileCreated};
use agentsmith_common::model::common::{Page, PagedResult};
use crate::model::party::PartyCreated;
use crate::model::person::{Person};
use crate::service::account::create_profile::create_profile_with_reference;
use crate::service::account::get_account::get_account_by_reference;
use crate::service::account::get_profile::get_profile_by_reference;
use crate::service::people::get_person::get_person_by_id;

#[derive(Clone)]
pub struct ProfileService {
    redis_pool: Arc<RedisPool>,
    database_pool: MySqlPool,
}

impl ProfileService {
    pub fn new(redis_pool: RedisPool, database_pool: MySqlPool) -> ProfileService {
        ProfileService { redis_pool: Arc::new(redis_pool), database_pool: database_pool }
    }
}

impl ProfileService {

    pub async fn create(&self, create_profile: CreateProfile, identifier: String) -> WebResult<Json<Profile>> {

        match get_profile_by_reference("oauth".to_string(), &identifier, "".to_string(), Utc::now(), &self.database_pool, &self.redis_pool).await {
            Some(profile) => {
                warn!("Profile already exists for {}", profile.id);
                Err(WebError::PersonError { id: 99, code: 99 })
            },
            _ => {
                create_profile_with_reference(create_profile, &identifier, &self.database_pool, &self.redis_pool).await?;
                match get_profile_by_reference("oauth".to_string(), &identifier, "".to_string(), Utc::now(), &self.database_pool, &self.redis_pool).await {
                    Some(profile) => Ok(Json(profile)),
                    _ => Err(WebError::PersonError { id: 99, code: 99 }),
                }
            }
        }
    }

    pub async fn list(&self, ) -> WebResult<Json<PagedResult<Account>>> {
        Ok(Json(PagedResult { page: Page {page: 0, size: 0, total:0 }, items: vec![] }))
    }

    pub async fn get(&self, reference: String, tenant: String) -> WebResult<Json<Profile>> {

        match get_profile_by_reference("oauth".to_string(), &reference, tenant, Utc::now(), &self.database_pool, &self.redis_pool).await {
            Some(profile) => Ok(Json(profile)),
            _ => Err(WebError::PersonError {id: 99, code: 99}),
        }
    }
}