use std::sync::Arc;
use axum::http::StatusCode;
use axum::Json;
use chrono::{DateTime, Utc};
use log::{debug, error, info};
use sqlx::{MySql, MySqlPool};
use sqlx::mysql::MySqlRow;
use sqlx::Row;
use agentsmith_common::database::sql::increment_counter;
use agentsmith_common::redis::common::RedisPool;
use crate::common::error::{WebError, WebResult as LResult };
use crate::model::account::{Account, CreateProfile, Profile};
use agentsmith_common::model::common::IdResult;
use crate::model::person::Person;

pub async fn create_profile_with_reference(create_profile: CreateProfile, identifier: &String, database_pool: &MySqlPool, redis_pool: &Arc<RedisPool>) -> LResult<u64> {

    debug!("Trying to create profile with identifier: {}", identifier);
    let id = increment_counter(0, database_pool, 0).await
        .map_err(|e| {
            error!("Error committing transaction: {}", e);
            WebError::AccountError { id: 1, code: 0 }
        })?;
    let mut transaction = database_pool.begin().await.map_err(|e| {
        error!("Error committing transaction: {}", e);
        WebError::AccountError { id: 0, code: 0 }
    })?;

    let _ = _create_profile(&create_profile, database_pool, id).await?;
    let _ = _create_profile_email(&create_profile, database_pool, id).await?;
    let _ = _create_profile_external_reference(identifier, database_pool, id).await?;

    transaction.commit().await.map_err(|e| {
        error!("Error committing transaction: {}", e);
        WebError::AccountError { id: 1, code: 0 }
    })?;
    Ok(id)
}

async fn _create_profile(create_profile: &CreateProfile, database_pool: &MySqlPool, id: u64) -> LResult<()> {

    let given_name = create_profile.given_name.clone();
    let family_name = create_profile.family_name.clone();

    sqlx::query(r#"INSERT INTO agentsmith.profile
                (id,
                eff_from,
                eff_to,
                name_prefix,
                name_suffix,
                givenname,
                middlename,
                familyname,
                known_as,
                date_of_birth,
                sex_at_birth,
                deceased_date,
                data,
                created_on,
                created_by,
                modified_on,
                modified_by,
                version,
                tenant_id)
            VALUES
                (?,
                CURRENT_TIMESTAMP,
                '9999-12-31 23:59:59',
                '',
                '',
                ?,
                null,
                ?,
                ?,
                null,
                'U',
                null,
                '{}',
                CURRENT_TIMESTAMP,
                CURRENT_USER,
                CURRENT_TIMESTAMP,
                CURRENT_USER,
                0,
                ''
                );"#)
        .bind(id)
        .bind(given_name.clone())
        .bind(family_name)
        .bind(given_name)
        .execute(database_pool)
        .await.map_err(|err: sqlx::Error| {
        error!("Error getting profile by reference: {}", err);
        debug!("Error getting profile by reference: {}", err);
        WebError::AccountError { id: 3, code: 0 }
    })?;
    Ok(())
}


async fn _create_profile_email(create_profile: &CreateProfile, database_pool: &MySqlPool, profile_id: u64) -> LResult<()> {

    let id = increment_counter(1, database_pool, 0).await.map_err(|err| {
        error!("Error getting profile by reference: {}", err);
        WebError::AccountError { id: 3, code: 3 }
    })?;

    sqlx::query(r#"INSERT INTO agentsmith.profile_email (id, profile_id, eff_from, eff_to, email, `order`, created_on, created_by, modified_on, modified_by, version, tenant_id)
        VALUES
        (?, ?, CURRENT_TIMESTAMP, '9999-12-31 23:59:59', ?, 0, CURRENT_TIMESTAMP, CURRENT_USER, CURRENT_TIMESTAMP, CURRENT_USER, 0, '');
"#)
        .bind(id)
        .bind(profile_id)
        .bind(create_profile.email.clone())
        .execute(database_pool)
        .await.map_err(|err: sqlx::Error| {
        error!("Error getting profile by reference: {}", err);
        debug!("Error getting profile by reference: {}", err);
        WebError::AccountError { id: 3, code: 3 }
    })?;

    Ok(())
}

async fn _create_profile_external_reference(reference: &String, database_pool: &MySqlPool, profile_id: u64) -> LResult<()> {

    let id = increment_counter(2, database_pool, 2).await.map_err(|err| {
        error!("Error getting profile by reference: {}", err);
        debug!("Error getting profile by reference: {}", err);
        WebError::AccountError { id: 3, code: 3 }
    })?;

    sqlx::query(r#"INSERT INTO agentsmith.profile_external_reference (id, profile_id, eff_from, eff_to, `system`, reference, created_on, created_by, modified_on, modified_by, version, tenant_id)
    VALUES
    (?, ?, CURRENT_TIMESTAMP, '9999-12-31 23:59:59', 'oauth', ?, CURRENT_TIMESTAMP, CURRENT_USER, CURRENT_TIMESTAMP, CURRENT_USER, 0, '');
"#)
        .bind(id)
        .bind(profile_id)
        .bind(reference)
        .execute(database_pool)
        .await.map_err(|err: sqlx::Error| {
        error!("Error getting profile by reference: {}", err);
        debug!("Error getting profile by reference: {}", err);
        WebError::AccountError { id: 3, code: 3 }
    })?;

    Ok(())
}


