use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartyCreated {
    #[serde(rename = "id")]
    pub id: i64,
}



#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Party {
    // Define your configuration structure
    #[serde(rename = "id")]
    pub id: i64,
    #[serde(rename = "type")]
    pub _type: String,
    #[serde(rename = "status")]
    pub status: String,
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
    #[serde(rename = "issuer")]
    pub version: i32,
}
