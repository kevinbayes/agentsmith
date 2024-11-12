use std::sync::Arc;
use axum::http::StatusCode;
use axum::Json;
use log::{debug, error, info};
use sqlx::{MySql, MySqlPool};
use sqlx::mysql::MySqlRow;
use agentsmith_common::redis::common::RedisPool;
use crate::common::error::WebError;
use crate::model::person::Person;

pub async fn get_person_by_id(id: String, tenant: String, database_pool: &MySqlPool, redis_pool: &Arc<RedisPool>) -> Option<Person> {

    info!("Trying to get person by id: {}", id);

    let query_result: Result<Vec<Person>, _> = sqlx::query_as(r#"select * from `agentsmith`.`person` p where p.id = ? and p.tenant_id = ?"#)
        .bind(id)
        .bind(tenant)
        .fetch_all(database_pool)
        .await
        .map_err(|err: sqlx::Error| {
            error!("Error getting person by id: {}", err);
            debug!("Error getting person by id: {}", err);
            WebError::PersonError {id: 0, code: 0}
        });

    match query_result {
        Ok(result) => {
            if result.len() > 0 {
                result.get(0).cloned()
            } else {
                None
            }
        },
        Err(_e) => {
            None
        }
    }
}