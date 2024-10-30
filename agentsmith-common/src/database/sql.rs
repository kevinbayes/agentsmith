use async_trait::async_trait;
use axum::extract::{FromRef, FromRequestParts};
use axum::http::request::Parts;
use axum::http::StatusCode;
use log::{debug, error};
use sqlx::{MySql, MySqlPool, Pool};
use sqlx::mysql::{MySqlPoolOptions, MySqlRow};
use sqlx::Row;
use crate::config::config::DatabaseConfig;
use crate::error::error::SystemError;
use crate::model::common::IdResult;

pub struct DatabaseConnection(sqlx::pool::PoolConnection<MySql>);

#[async_trait]
impl<S> FromRequestParts<S> for DatabaseConnection
    where
        MySqlPool: FromRef<S>,
        S: Send + Sync,
{
    type Rejection = (StatusCode, String);

    async fn from_request_parts(_parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let pool = MySqlPool::from_ref(state);

        let conn = pool.acquire().await.map_err(internal_error)?;

        Ok(Self(conn))
    }
}


pub async fn create_mysql_pool(database_config: DatabaseConfig) -> MySqlPool {
    match MySqlPoolOptions::new()
        .max_connections(10)
        .connect(database_config.connection.as_str())
        .await
    {
        Ok(pool) => {
            println!("✅ Connection to the database is successful!");
            MySqlPool::from(pool)
        }
        Err(err) => {
            println!("❌ Failed to connect to the database: {:?}", err);
            std::process::exit(1);
        }
    }
}

fn internal_error<E>(err: E) -> (StatusCode, String)
    where
        E: std::error::Error,
{
    (StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
}



pub async fn increment_counter(counter: u32, database_pool: &MySqlPool, correlation: u8) -> crate::error::error::SystemResult<u64> {

    let increment_result = sqlx::query(r#"CALL agentsmith.get_next_id(?)"#)
        .bind(counter)
        .map(|row: MySqlRow| {
            IdResult {
                id: row.get(0),
            }
        })
        .fetch_all(database_pool)
        .await.map_err(|err: sqlx::Error| {
        error!("Error getting profile by reference: {}", err);
        debug!("Error getting profile by reference: {}", err);
        SystemError::IncrementNumberFailed  { id: correlation }
    });

    match increment_result {
        Ok(_result) => {
            if _result.len() > 0 {
                Ok(_result.get(0).cloned().unwrap().id)
            } else {
                Err(SystemError::IncrementNumberFailed { id: correlation })
            }
        },
        Err(e) => {
            Err(SystemError::IncrementNumberFailed { id: correlation })
        }
    }
}