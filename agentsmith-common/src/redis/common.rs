use r2d2_redis::{r2d2, RedisConnectionManager};
use crate::config::config::{RedisConfig};

pub type RedisPool = r2d2::Pool<RedisConnectionManager>;

//TODO: Define custom error cause
pub fn create_redis_pool(redis_config: RedisConfig) -> Result<RedisPool, String> {
    let redis_url = format!("redis://{}/", redis_config.host);

    let manager = RedisConnectionManager::new(redis_url)
        .map_err(|e| format!("Error: {}", e.to_string()))?;

    let pool = r2d2::Pool::builder()
        .max_size(redis_config.connection_pool_size)
        .build(manager)
        .map_err(|e| format!("Error: {}", e.to_string()))?;

    Ok(pool)
}