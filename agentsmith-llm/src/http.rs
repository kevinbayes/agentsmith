//! Shared chat HTTP: one POST at a time, the same retry table for every provider.
//!
//! 429, 5xx, transport failures, and timeouts wait 500ms, then 1500ms, with
//! ±20% jitter. 401 and 403 are `Auth` and are not retried. Other 4xx are
//! `BadRequest` and are not retried. Response bodies are dropped on errors.

use std::time::Duration;

use agentsmith_core::{Error, LlmErrorKind};
use reqwest::StatusCode;
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use url::Url;

/// Extra attempts after the first. Matches the config ceiling in `agentsmith-core`.
pub(crate) const MAX_RETRIES: u32 = 4;
const BACKOFF_FIRST_MS: u64 = 500;
const BACKOFF_LATER_MS: u64 = 1500;

pub(crate) fn http_base(base_url: &str) -> Result<Url, Error> {
    let url = Url::parse(base_url).map_err(|err| Error::Config(format!("llm.base_url: {err}")))?;
    match url.scheme() {
        "http" | "https" => Ok(url),
        other => Err(Error::Config(format!(
            "llm.base_url scheme {other} is not supported"
        ))),
    }
}

pub(crate) fn llm_error(kind: LlmErrorKind, message: impl Into<String>) -> Error {
    Error::Llm {
        kind,
        message: message.into(),
    }
}

/// POST JSON and return the success body.
///
/// `bearer` is sent as `Authorization: Bearer`. `headers` are extra names such
/// as `x-api-key`. Neither value is copied into an error.
pub(crate) async fn post_json(
    client: &reqwest::Client,
    retries: u32,
    timeout: Duration,
    url: &Url,
    bearer: Option<&str>,
    headers: &[(&str, &str)],
    body: &Value,
    cancel: &CancellationToken,
) -> Result<String, Error> {
    if cancel.is_cancelled() {
        return Err(Error::Cancelled);
    }
    let max_attempts = retries + 1;
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        match once(client, timeout, url, bearer, headers, body, cancel).await {
            Ok(text) => return Ok(text),
            Err(OnceError::Cancelled) => return Err(Error::Cancelled),
            Err(OnceError::Done(err)) => return Err(err),
            Err(OnceError::Retry(err)) => {
                if attempt >= max_attempts {
                    return Err(err);
                }
                let delay = backoff_delay(attempt - 1);
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return Err(Error::Cancelled),
                    _ = tokio::time::sleep(delay) => {}
                }
            }
        }
    }
}

enum OnceError {
    Cancelled,
    Done(Error),
    Retry(Error),
}

async fn once(
    client: &reqwest::Client,
    timeout: Duration,
    url: &Url,
    bearer: Option<&str>,
    headers: &[(&str, &str)],
    body: &Value,
    cancel: &CancellationToken,
) -> Result<String, OnceError> {
    if cancel.is_cancelled() {
        return Err(OnceError::Cancelled);
    }

    let mut pending = client.post(url.clone()).timeout(timeout);
    if let Some(token) = bearer {
        pending = pending.bearer_auth(token);
    }
    for (name, value) in headers {
        pending = pending.header(*name, *value);
    }
    let pending = pending.json(body).send();

    let response = tokio::select! {
        biased;
        _ = cancel.cancelled() => return Err(OnceError::Cancelled),
        result = pending => result,
    };
    let response = match response {
        Ok(response) => response,
        Err(err) => return Err(transport_error(&err)),
    };

    let status = response.status();
    let body_result = tokio::select! {
        biased;
        _ = cancel.cancelled() => return Err(OnceError::Cancelled),
        result = response.text() => result,
    };

    if !status.is_success() {
        drop(body_result);
        let (kind, retry) = classify_status(status);
        let err = llm_error(kind, format!("HTTP {status}"));
        return Err(if retry {
            OnceError::Retry(err)
        } else {
            OnceError::Done(err)
        });
    }

    match body_result {
        Ok(text) => Ok(text),
        Err(err) => Err(transport_error(&err)),
    }
}

fn classify_status(status: StatusCode) -> (LlmErrorKind, bool) {
    match status.as_u16() {
        401 | 403 => (LlmErrorKind::Auth, false),
        429 => (LlmErrorKind::Unavailable, true),
        500..=599 => (LlmErrorKind::Unavailable, true),
        400..=499 => (LlmErrorKind::BadRequest, false),
        _ => (LlmErrorKind::Unavailable, false),
    }
}

fn transport_error(err: &reqwest::Error) -> OnceError {
    if err.is_timeout() {
        OnceError::Retry(llm_error(LlmErrorKind::Timeout, "request timed out"))
    } else if err.is_builder() {
        OnceError::Done(llm_error(LlmErrorKind::BadRequest, "invalid HTTP request"))
    } else {
        OnceError::Retry(llm_error(
            LlmErrorKind::Unavailable,
            format!("request failed: {err}"),
        ))
    }
}

pub(crate) fn backoff_delay(retry_index: u32) -> Duration {
    let base = if retry_index == 0 {
        BACKOFF_FIRST_MS
    } else {
        BACKOFF_LATER_MS
    };
    let min = base.saturating_mul(8) / 10;
    let max = (base.saturating_mul(12) / 10).max(min);
    Duration::from_millis(rand::random_range(min..=max))
}
