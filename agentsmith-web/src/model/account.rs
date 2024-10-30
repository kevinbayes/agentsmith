use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use log::debug;
use sqlx::{Error, FromRow};
use sqlx::mysql::MySqlRow;
use sqlx::Row;
use agentsmith_common::model::common::Name;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Account {
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

impl FromRow<'_, MySqlRow> for Account {

    fn from_row(row: &MySqlRow) -> sqlx::Result<Self> {

        let id = row.try_get("id")?;
        debug!("Mapping account: {}.", id);

        Ok(Self {
            id,
            name: row.try_get("name")?,
            _type: row.try_get("type")?,
            status: row.try_get("status")?,
            tenant_id: row.try_get("tenant_id")?,
            created_on: row.try_get("created_on")?,
            created_by: row.try_get("created_by")?,
            modified_on: row.try_get("modified_on")?,
            modified_by: row.try_get("modified_by")?,
            version: row.try_get("version")?,
        })
    }
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateProfile {
    // Define your configuration structure
    #[serde(rename = "given_name")]
    pub given_name: String,
    #[serde(rename = "family_name")]
    pub family_name: String,
    #[serde(rename = "email")]
    pub email: String,
    #[serde(rename = "email_consent")]
    pub email_consent: bool,
    #[serde(rename = "terms")]
    pub terms: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileCreated {
    #[serde(rename = "id")]
    pub id: i64,
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Profile {
    // Define your configuration structure
    #[serde(rename = "id")]
    pub id: u64,
    #[serde(rename = "name")]
    pub name: Name,
    #[serde(rename = "eff_from")]
    pub eff_from: DateTime<Utc>,
    #[serde(rename = "eff_to")]
    pub eff_to: DateTime<Utc>,
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

impl FromRow<'_, MySqlRow> for Profile {

    fn from_row(row: &MySqlRow) -> sqlx::Result<Self> {

        let id = row.try_get("id")?;
        debug!("Mapping profile: {}.", id);

        Ok(Self {
            id,
            name: Name {
                given: row.try_get("givenname")?,
                middle: row.try_get("middlename")?,
                family: row.try_get("familyname")?,
            },
            eff_from: row.try_get("eff_from")?,
            eff_to: row.try_get("eff_to")?,
            tenant_id: row.try_get("tenant_id")?,
            created_on: row.try_get("created_on")?,
            created_by: row.try_get("created_by")?,
            modified_on: row.try_get("modified_on")?,
            modified_by: row.try_get("modified_by")?,
            version: row.try_get("version")?,
        })
    }
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileEmail {
    // Define your configuration structure
    #[serde(rename = "id")]
    pub id: u64,
    #[serde(rename = "id")]
    pub profile_id: u64,
    #[serde(rename = "email")]
    pub email: String,
    #[serde(rename = "order")]
    pub order: u8,
    #[serde(rename = "eff_from")]
    pub eff_from: DateTime<Utc>,
    #[serde(rename = "eff_to")]
    pub eff_to: DateTime<Utc>,
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

impl FromRow<'_, MySqlRow> for ProfileEmail {

    fn from_row(row: &MySqlRow) -> sqlx::Result<Self> {

        let id = row.try_get("id")?;
        debug!("Mapping profile: {}.", id);

        Ok(Self {
            id,
            profile_id: row.try_get("profile_id")?,
            email: row.try_get("email")?,
            order: row.try_get("order")?,
            eff_from: row.try_get("eff_from")?,
            eff_to: row.try_get("eff_to")?,
            tenant_id: row.try_get("tenant_id")?,
            created_on: row.try_get("created_on")?,
            created_by: row.try_get("created_by")?,
            modified_on: row.try_get("modified_on")?,
            modified_by: row.try_get("modified_by")?,
            version: row.try_get("version")?,
        })
    }
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileExternalReference {
    // Define your configuration structure
    #[serde(rename = "id")]
    pub id: u64,
    #[serde(rename = "id")]
    pub profile_id: u64,
    #[serde(rename = "system")]
    pub system: String,
    #[serde(rename = "reference")]
    pub reference: String,
    #[serde(rename = "eff_from")]
    pub eff_from: DateTime<Utc>,
    #[serde(rename = "eff_to")]
    pub eff_to: DateTime<Utc>,
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

impl FromRow<'_, MySqlRow> for ProfileExternalReference {

    fn from_row(row: &MySqlRow) -> sqlx::Result<Self> {

        let id = row.try_get("id")?;
        debug!("Mapping profile: {}.", id);

        Ok(Self {
            id,
            profile_id: row.try_get("profile_id")?,
            system: row.try_get("system")?,
            reference: row.try_get("reference")?,
            eff_from: row.try_get("eff_from")?,
            eff_to: row.try_get("eff_to")?,
            tenant_id: row.try_get("tenant_id")?,
            created_on: row.try_get("created_on")?,
            created_by: row.try_get("created_by")?,
            modified_on: row.try_get("modified_on")?,
            modified_by: row.try_get("modified_by")?,
            version: row.try_get("version")?,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccountUsers {
    // Define your configuration structure
    #[serde(rename = "id")]
    pub id: u64,
    #[serde(rename = "profile_id")]
    pub profile_id: u64,
    // #[serde(rename = "data")]
    // #[sqlx::convert = Name]
    // pub data: Name,
    #[serde(rename = "account_id")]
    pub account_id: u64,
    #[serde(rename = "type")]
    pub status: i8,
    #[serde(rename = "eff_from")]
    pub eff_from: DateTime<Utc>,
    #[serde(rename = "eff_to")]
    pub eff_to: DateTime<Utc>,
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
    #[serde(rename = "tenant_id")]
    pub tenant_id: String,
}

impl FromRow<'_, MySqlRow> for AccountUsers {

    fn from_row(row: &MySqlRow) -> sqlx::Result<Self> {

        let id = row.try_get("id")?;
        debug!("Mapping account users: {}.", id);

        Ok(Self {
            id,
            profile_id: row.try_get("profile_id")?,
            account_id: row.try_get("account_id")?,
            status: row.try_get("status")?,
            eff_from: row.try_get("eff_from")?,
            eff_to: row.try_get("eff_to")?,
            tenant_id: row.try_get("tenant_id")?,
            created_on: row.try_get("created_on")?,
            created_by: row.try_get("created_by")?,
            modified_on: row.try_get("modified_on")?,
            modified_by: row.try_get("modified_by")?,
            version: row.try_get("version")?,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccountUsersRole {
    // Define your configuration structure
    #[serde(rename = "id")]
    pub id: u64,
    #[serde(rename = "account_user_id")]
    pub account_user_id: u64,
    // #[serde(rename = "data")]
    // #[sqlx::convert = Name]
    // pub data: Name,
    #[serde(rename = "role")]
    pub role: String,
    #[serde(rename = "type")]
    pub status: i8,
    #[serde(rename = "eff_from")]
    pub eff_from: DateTime<Utc>,
    #[serde(rename = "eff_to")]
    pub eff_to: DateTime<Utc>,
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
    #[serde(rename = "tenant_id")]
    pub tenant_id: String,
}

impl FromRow<'_, MySqlRow> for AccountUsersRole {

    fn from_row(row: &MySqlRow) -> sqlx::Result<Self> {

        let id = row.try_get("id")?;
        debug!("Mapping account users role: {}.", id);

        Ok(Self {
            id,
            account_user_id: row.try_get("account_user_id")?,
            role: row.try_get("role")?,
            status: row.try_get("status")?,
            eff_from: row.try_get("eff_from")?,
            eff_to: row.try_get("eff_to")?,
            tenant_id: row.try_get("tenant_id")?,
            created_on: row.try_get("created_on")?,
            created_by: row.try_get("created_by")?,
            modified_on: row.try_get("modified_on")?,
            modified_by: row.try_get("modified_by")?,
            version: row.try_get("version")?,
        })
    }
}