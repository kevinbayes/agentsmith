use log::debug;
use serde::{Deserialize, Serialize};
use sqlx::{Error, FromRow};
use sqlx::mysql::MySqlRow;
use sqlx::Row;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdResult {
    pub id: u64,
}


impl FromRow<'_, MySqlRow> for IdResult {

    fn from_row(row: &MySqlRow) -> sqlx::Result<Self> {

        let id = row.try_get("id")?;
        debug!("Mapping id: {}.", id);

        Ok(Self {
            id,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Page {
    pub page: i32,
    pub size: i32,
    pub total: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PagedResult<T> {
    pub page: Page,
    pub items: Vec<T>,
}

impl<T> PagedResult<T> {
    pub fn new(page: Page, items: Vec<T>) -> Self {
        PagedResult { page, items }
    }
}


pub mod standard_date_format {
    use chrono::{DateTime, Utc, NaiveDateTime};
    use serde::{self, Deserialize, Serializer, Deserializer};

    const FORMAT: &'static str = "%Y-%m-%d %H:%M:%SZ";

    // The signature of a serialize_with function must follow the pattern:
    //
    //    fn serialize<S>(&T, S) -> Result<S::Ok, S::Error>
    //    where
    //        S: Serializer
    //
    // although it may also be generic over the input types T.
    pub fn serialize<S>(
        date: &DateTime<Utc>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
    {
        let s = format!("{}", date.format(FORMAT));
        serializer.serialize_str(&s)
    }

    // The signature of a deserialize_with function must follow the pattern:
    //
    //    fn deserialize<'de, D>(D) -> Result<T, D::Error>
    //    where
    //        D: Deserializer<'de>
    //
    // although it may also be generic over the output types T.
    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<DateTime<Utc>, D::Error>
        where
            D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let dt = NaiveDateTime::parse_from_str(&s, FORMAT).map_err(serde::de::Error::custom)?;
        Ok(DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Name {
    #[serde(rename = "given")]
    pub given: String,
    #[serde(rename = "middle")]
    pub middle: Option<String>,
    #[serde(rename = "family")]
    pub family: String,
}