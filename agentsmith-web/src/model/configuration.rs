use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Configuration {
    // Define your configuration structure
    #[serde(rename = "id")]
    pub id: u64,
    #[serde(rename = "name")]
    pub name: String,
    #[serde(rename = "type")]
    pub _type: String,
    #[serde(rename = "type")]
    pub status: i8,
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
