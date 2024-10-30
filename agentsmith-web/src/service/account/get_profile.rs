use std::sync::Arc;
use axum::http::StatusCode;
use axum::Json;
use chrono::{DateTime, Utc};
use log::{debug, error, info};
use sqlx::{MySql, MySqlPool};
use sqlx::mysql::MySqlRow;
use agentsmith_common::redis::common::RedisPool;
use crate::common::error::WebError;
use crate::model::account::{Account, Profile};
use crate::model::person::Person;

pub async fn get_profile_by_reference(system: String, reference: &String, tenant: String, at: DateTime<Utc>, database_pool: &MySqlPool, redis_pool: &Arc<RedisPool>) -> Option<Profile> {

    info!("Trying to get account by reference: {}/{}", system, reference);

    let query_result: Result<Vec<Profile>, _> = sqlx::query_as(r#"select * from `agentsmith`.`profile` p where p.id in ( select per.profile_id from `agentsmith`.`profile_external_reference` per where per.system = ? and per.reference = ? and per.tenant_id = ? and ? between per.eff_from and per.eff_to )"#)
        .bind(system)
        .bind(reference)
        .bind(tenant)
        .bind(at)
        .fetch_all(database_pool)
        .await
        .map_err(|err: sqlx::Error| {
            error!("Error getting profile by reference: {}", err);
            debug!("Error getting profile by reference: {}", err);
            WebError::AccountError {id: 0, code: 0}
        });

    match query_result {
        Ok(result) => {
            if result.len() > 0 {
                result.get(0).cloned()
            } else {
                None
            }
        },
        Err(e) => {
            None
        }
    }
}

