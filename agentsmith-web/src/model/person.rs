use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use log::debug;
use sqlx::{Error, FromRow};
use sqlx::mysql::MySqlRow;
use sqlx::Row;
use agentsmith_common::model::common::Name;


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Person {
    // Define your configuration structure
    #[serde(rename = "id")]
    pub id: u64,
    #[serde(rename = "name")]
    pub name: Name,
    // #[serde(rename = "data")]
    // #[sqlx::convert = Name]
    // pub data: Name,
    #[serde(rename = "tenant_id")]
    pub tenant_id: String,
    #[serde(rename = "created_on")]
    pub created_on: DateTime<Utc>,
    #[serde(rename = "created_by")]
    pub created_by: String,
    #[serde(rename = "modified_on")]
    pub modified_on: DateTime<Utc>,
    #[serde(rename = "modified_by")]
    pub modified_by: String,
    #[serde(rename = "version")]
    pub version: i32,
}

impl FromRow<'_, MySqlRow> for Person {

    fn from_row(row: &MySqlRow) -> sqlx::Result<Self> {

        let id = row.try_get("id")?;
        debug!("Mapping person: {}.", id);

        Ok(Self {
            id,
            name: Name {
                given: row.try_get("givenname")?,
                middle: row.try_get("middlename")?,
                family: row.try_get("familyname")?,
            },
            tenant_id: row.try_get("tenant_id")?,
            created_on: row.try_get("created_on")?,
            created_by: row.try_get("created_by")?,
            modified_on: row.try_get("modified_on")?,
            modified_by: row.try_get("modified_by")?,
            version: row.try_get("version")?,
        })
    }
}