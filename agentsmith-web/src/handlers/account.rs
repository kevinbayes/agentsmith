use crate::config::config::Config;
use crate::handlers::auth_middleware::Claims;
use crate::model::account::{CreateProfile, Profile};
use crate::service::account_service::AccountService;
use crate::service::profile_service::ProfileService;
use axum::extract::{Extension, FromRef, State};
use axum::http::HeaderMap;
use axum::routing::get;
use axum::{Json, Router};
use crate::common::error::WebResult;

#[derive(Clone, FromRef)]
struct AccountState {
    service: AccountService,
}

#[derive(Clone, FromRef)]
struct ProfileState {
    service: ProfileService,
}

pub fn account_routes (account_service: AccountService, profile_service: ProfileService, config: Config) -> Router {
    let profile_state   = ProfileState { service: profile_service };

    let profiles_routes = Router::new()
        .route("/me", get(profile_handler).post(profile_create_handler))
        .with_state(profile_state);

    Router::new()
        .nest("/api/profiles", profiles_routes)
}

pub async fn profile_handler(
    headers: HeaderMap,
    State(state): State<ProfileState>,
    Extension(claims): Extension<Claims>,
) -> WebResult<Json<Profile>> {
    let user = claims.sub;
    state.service.get(user, "".to_string()).await
}

pub async fn profile_create_handler(
    headers: HeaderMap,
    State(state): State<ProfileState>,
    Extension(claims): Extension<Claims>,
    Json(payload): Json<CreateProfile>,
) -> WebResult<Json<Profile>> {
    let user = claims.sub;
    state.service.create(payload, user).await
}
