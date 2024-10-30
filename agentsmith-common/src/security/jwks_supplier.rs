use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration as StdDuration;

use chrono::{Duration, Utc};
use tokio::sync::RwLock;
use tokio::sync::{RwLockReadGuard, RwLockWriteGuard};

use jsonwebtoken::{
    decode, decode_header,
    jwk::{AlgorithmParameters, JwkSet},
    Algorithm, DecodingKey, Validation,
};
use jsonwebtoken::jwk::Jwk;
use tracing::error;
use crate::config::config::{Config, OAuthConfig};
use crate::error::error::SystemError;

#[derive(Clone)]
pub struct JwksReadThroughCache {
    uri: Arc<String>,
    inner: Arc<RwLock<Entry>>,
    time_to_live: Duration,
    refreshed: Arc<AtomicBool>,
    static_jwks: Option<String>,
}

impl JwksReadThroughCache {

    pub fn new(config: OAuthConfig, time_to_live: StdDuration, static_jwks: Option<String>) -> Self {

        let domain = config.jwks_domain;
        let protocol = config.jwks_protocol;
        let path = config.jwks_path;

        let uri = format!("{}://{}/{}", protocol, domain, path);

        let ttl: Duration = Duration::from_std(time_to_live)
            .expect("Failed to convert from `std::time::Duration` to `chrono::Duration`");
        let json_web_key_set: JwkSet = JwkSet { keys: vec![], };

        Self {
            uri: Arc::new(uri),
            inner: Arc::new(RwLock::new(Entry::new(json_web_key_set, &ttl))),
            time_to_live: ttl,
            refreshed: Arc::new(AtomicBool::new(false)),
            static_jwks,
        }
    }

    pub async fn get_or_refresh(
        &self,
        key: &str,
    ) -> Result<Jwk, SystemError> {

        let read: RwLockReadGuard<Entry> = self.inner.read().await;
        let is_entry_expired: bool = read.is_expired();
        let get_key_result: Option<Jwk> = read.set.find(key).cloned();

        drop(read);

        match get_key_result {
            Some(jwk) if !is_entry_expired => Ok(jwk),
            _ => self.try_refresh().await.and_then(|v| v.find(key).cloned().ok_or(SystemError::JwksError)),
        }
    }

    async fn try_refresh(&self) -> Result<JwkSet, SystemError> {
        self.refreshed.store(false, Ordering::SeqCst);
        let mut guard: RwLockWriteGuard<Entry> = self.inner.write().await;

        if !self.refreshed.load(Ordering::SeqCst) {
            let set: JwkSet = self.get_jwkset().await?;
            *guard = Entry::new(set.clone(), &self.time_to_live);
            self.refreshed.store(true, Ordering::SeqCst);
            Ok(set)
        } else {
            Ok(guard.set.clone())
        }
    }

    async fn get_jwkset(&self) -> Result<JwkSet, SystemError> {

        let jwks_str = self.static_jwks.clone();

        match jwks_str {
            Some(jwks) => self.get_static_jwks(jwks),
            _ => self.call_jwks().await,
        }
    }

    fn get_static_jwks(&self, jwks: String) -> Result<JwkSet, SystemError> {

        Ok(serde_json::from_str::<JwkSet>(jwks.as_str()).unwrap())
    }


    async fn call_jwks(&self) -> Result<JwkSet, SystemError> {

        let uri = self.uri.clone();

        return match reqwest::get(format!("{}", uri)).await {
            Ok(response) => {
                if response.status() != 200 {
                    error!("1. Error with message deserialising {:?}", response);
                    return Err(SystemError::AuthFailed)
                }
                match response.json::<JwkSet>().await {
                    Ok(jwks) => {
                        println!("->> {:?} - got it", jwks.keys);
                        Ok(jwks)
                    },
                    Err(e) => {
                        error!("2. Error with message deserialising {:?}", e);
                        Err(SystemError::AuthFailed)
                    },
                }
            }
            Err(e) => {
                error!("3. Error with message deserialising {:?}", e);
                Err(SystemError::AuthFailed)
            }
        }
    }
}


struct Entry {
    set: JwkSet,
    expire_time_millis: i64,
}

impl Entry {
    fn new(set: JwkSet, expiration: &Duration) -> Self {
        Self {
            set,
            expire_time_millis: Utc::now().timestamp_millis() + expiration.num_milliseconds(),
        }
    }

    fn is_expired(&self) -> bool {
        Utc::now().timestamp_millis() > self.expire_time_millis
    }
}


#[cfg(test)]
mod tests {
    use std::fs;
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_read_jwks() {

        let jwks_str = fs::read_to_string("./src/common/jwks.mock.response.json").unwrap();

        let cache = JwksReadThroughCache::new(
            OAuthConfig {
                jwks_protocol:"".to_string(),
                jwks_domain: "".to_string(),
                jwks_path: "".to_string(),
                audience: "".to_string()
            }, StdDuration::from_secs(3600), Some(jwks_str)
        );

        let jwk1 = cache.get_or_refresh("8mda2DcaodWkXQZPnWskY").await.expect("jwks1");
        assert_eq!(jwk1.common.key_id.unwrap(), "8mda2DcaodWkXQZPnWskY");

        let jwk2 = cache.get_or_refresh("UQIOC2xRgO7kWitPQUm8E").await.expect("jwks2");
        assert_eq!(jwk2.common.key_id.unwrap(), "UQIOC2xRgO7kWitPQUm8E");

        let jwk3 = cache.get_or_refresh("UQIOC2xRgO7kWitPQUm8E").await.expect("jwks3");
        assert_eq!(jwk3.common.key_id.unwrap(), "UQIOC2xRgO7kWitPQUm8E");
    }
}
