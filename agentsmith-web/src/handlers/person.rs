use crate::config::config::Config;
use crate::model::person::Person;
use crate::service::people_service::PersonService;
// use crate::common::error::Result;
use axum::extract::Query;
use axum::extract::{FromRef, Path, State};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use crate::common::error::WebResult;

#[derive(Clone, FromRef)]
struct PersonState {
    ps: PersonService,
}

pub fn person_routes(ps: PersonService, config: Config) -> Router {
    let state = PersonState { ps };

    Router::new()
        .route("/api/people/:id", get(person_handler))
        .with_state(state)
}

#[derive(Debug, Deserialize)]
pub struct PersonPathParams {
    id: String,
}

#[derive(Debug, Deserialize)]
pub struct PersonQueryParams {
    #[serde(rename = "i")]
    include: Option<String>,
}

pub async fn person_handler(State(state): State<PersonState>,
                            Path(path_params): Path<PersonPathParams>,
                            Query(query_params): Query<PersonQueryParams>) -> WebResult<Json<Person>> {
    let i = query_params.include.as_deref().unwrap_or("all");
    let id = path_params.id;
    let tenant = String::from("demo");
    state.ps.get(id, tenant).await
}
