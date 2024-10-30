mod common;
mod config;
mod gateway;
mod handlers;
mod model;
mod pb;
mod security;
mod service;
mod context;


use std::collections::HashSet;
use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;
use axum::{Json, middleware, Router};
use axum::http::{Method, Uri};
use axum::routing::get;
use axum::response::Response;
use axum::response::IntoResponse;
use serde_json::json;
use tower_http::services::ServeDir;
use tracing::info;
use sqlx::mysql::{MySqlPool, MySqlPoolOptions};
use sqlx::{MySql, Pool};
use uuid::Uuid;

use crate::config::config::{read_config, DatabaseConfig, RedisConfig};
use agentsmith_common::redis::common::create_redis_pool;
use agentsmith_common::database::sql::create_mysql_pool;
use agentsmith_common::security::jwks_supplier::JwksReadThroughCache;
use crate::context::Context;
use crate::common::error::WebError;
use crate::gateway::gateway_registry::create_gateway_registry;
use crate::handlers::{actuator::actuator_routes, person::person_routes};
use crate::handlers::account::account_routes;
use crate::handlers::auth_middleware::{auth_middleware, AuthState};
use crate::service::account_service::AccountService;
use crate::service::people_service::PersonService;
use crate::service::profile_service::ProfileService;



#[tokio::main]
async fn main() {

    let value = env::var("CONFIG_LOCATION").unwrap_or_else(|_| "./config.json".to_string());
    let configuration = &read_config(&value).unwrap();

    let redis_pool = create_redis_pool(configuration.config.redis.clone().into()).unwrap();
    let database_pool = create_mysql_pool(configuration.config.database.clone().into()).await;
    let gateway_registry = create_gateway_registry(configuration.clone()).await;

    let auth_config = configuration.clone();
    let oauth_config = configuration.config.security.oauth.clone();
    let auth_state = AuthState {
        config: auth_config,
        cache: JwksReadThroughCache::new(oauth_config.into(), Duration::from_secs(3600), None)
    };

    let secure_app = Router::new()
        .merge(account_routes(AccountService::new(redis_pool.clone(), database_pool.clone()), ProfileService::new(redis_pool.clone(), database_pool.clone()), configuration.clone()))
        .route_layer(middleware::from_fn_with_state(auth_state, auth_middleware));

    let app = Router::new()
        .merge(actuator_routes())
        .merge(secure_app)
        .layer(middleware::map_response(main_response_mapper))
        .fallback_service(static_routes())
        ;

    let addr = format!("{}:{}", configuration.config.host.host, configuration.config.host.port);
    let listener = tokio::net::TcpListener::bind(addr.as_str())
        .await
        .unwrap();
    tracing::warn!("listening on {}", addr.as_str());
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
        .await
        .unwrap();
}


async fn main_response_mapper(
    res: Response,
) -> Response {
    println!("->> {:<12} - main_response_mapper", "RES_MAPPER");

    let uuid = Uuid::new_v4();

    // -- Get the eventual response error.
    let service_error = res.extensions().get::<WebError>();
    let client_status_error = service_error.map(|se| se.client_status_and_error());

    let error_response =
        client_status_error
            .as_ref()
            .map(|(status_code, client_error)| {
                let client_error_body = json!({
    				"error": {
    					"type": client_error.as_ref(),
    					"req_uuid": uuid.to_string(),
    				}
    			});

                println!("    ->> client_error_body: {client_error_body}");

                // Build the new response from the client_error_body
                (*status_code, Json(client_error_body)).into_response()
            });

    error_response.unwrap_or(res)

    //
    // // -- If client error, build the new reponse.
    // let error_response =
    //     client_status_error
    //         .as_ref()
    //         .map(|(status_code, client_error)| {
    //             let client_error_body = json!({
	// 				"error": {
	// 					"type": client_error.as_ref(),
	// 					"req_uuid": uuid.to_string(),
	// 				}
	// 			});
    //
    //             println!("    ->> client_error_body: {client_error_body}");
    //
    //             // Build the new response from the client_error_body
    //             (*status_code, Json(client_error_body)).into_response()
    //         });
    //
    // // Build and log the server log line.
    // let client_error = client_status_error.unzip().1;
    // // TODO: Need to hander if log_request fail (but should not fail request)
    // let _ =
    //     log_request(uuid, req_method, uri, ctx, service_error, client_error).await;
    //
    // println!();
    // error_response.unwrap_or(res)
}

fn static_routes ()-> Router {
    Router::new().nest_service("/", ServeDir::new("assets"))
}

//