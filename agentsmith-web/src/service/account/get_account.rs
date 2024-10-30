use std::sync::Arc;
use axum::http::StatusCode;
use axum::Json;
use log::{debug, error, info};
use sqlx::{MySql, MySqlPool};
use sqlx::mysql::MySqlRow;
use agentsmith_common::redis::common::RedisPool;
use crate::common::error::WebError;
use crate::model::account::Account;
use crate::model::person::Person;

pub async fn get_account_by_reference(system: String, reference: String, tenant: String, database_pool: &MySqlPool, redis_pool: &Arc<RedisPool>) -> Option<Account> {

    info!("Trying to get account by reference: {}/{}", system, reference);

    let query_result: Result<Vec<Account>, _> = sqlx::query_as(r#"select * from `agentsmith`.`account` p where p.id = ? and p.tenant_id = ?"#)
        .bind(system)
        .bind(reference)
        .bind(tenant)
        .fetch_all(database_pool)
        .await
        .map_err(|err: sqlx::Error| {
            error!("Error getting account by reference: {}", err);
            debug!("Error getting person by reference: {}", err);
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