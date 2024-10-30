use std::env;
use std::future::Future;
use async_trait::async_trait;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::response::Response;
use crate::error::{Error, Result};
use axum::body::Body;
use axum::extract::{FromRequest, FromRequestParts, State};
use axum::http::header::AUTHORIZATION;
use axum::http::request::Parts;
use axum::middleware::Next;
use axum::{RequestExt, RequestPartsExt};
use axum_extra::TypedHeader;
use futures_util::TryFutureExt;
use headers::Authorization;
use headers::authorization::Bearer;
use reqwest;



use jsonwebtoken::{decode, decode_header, jwk::{AlgorithmParameters, JwkSet}, Algorithm, DecodingKey, Validation, TokenData};
use jsonwebtoken::jwk::Jwk;
use serde::Deserialize;
use tracing::error;
use crate::common::jwks_supplier::JwksReadThroughCache;
use crate::config::config::Config;
use crate::error::ClientError;

#[derive(Clone)]
pub struct AuthState {
    pub config: Config,
    pub cache: JwksReadThroughCache,
}

pub async fn auth_middleware(
    State(state): State<AuthState>,
    req: Request<Body>,
    next: Next,
) -> Result<Response> {
    println!("->> {:<12} - mw_require_auth", state.config.config.security.oauth.jwks_domain);

    let audience = state.config.config.security.oauth.audience.clone();

    let (mut parts, body) = req.into_parts();
    let TypedHeader(Authorization(bearer)) = parts.extract::<TypedHeader<Authorization<Bearer>>>()
        .await
        .map_err(|_| Error::InvalidToken)?;

    let token = bearer.token();
    let header = decode_header(token).map_err(|err| Error::InvalidToken)?;

    let kid = header.kid.ok_or_else(|| Error::InvalidToken)?;
    let algo = header.alg;

    let jwk = state.cache.get_or_refresh(kid.as_str()).await?;

    let decoding_key = DecodingKey::from_jwk(&jwk).unwrap();

    let mut validation = Validation::new(algo);
    validation.set_audience(&[audience]);

    let token_data = decode::<Claims>(bearer.token(), &decoding_key, &validation)
        .map_err(|e| {
            println!("Error with message deserialising {:?}", e);
            error!("Error with message deserialising {:?}", e);
            Error::InvalidToken
        })?;

    let mut req = Request::from_parts(parts, body);
    req.extensions_mut().insert(token_data.claims);

    Ok(next.run(req).await)
}

#[derive(Clone, Debug, Deserialize)]
pub struct Claims {
    pub sub: String,
}