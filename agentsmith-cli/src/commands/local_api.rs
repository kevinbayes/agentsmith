//! Loopback HTTP/1. No framework.

use std::path::Path;
use std::sync::Arc;

use agentsmith_core::redact::redact;
use agentsmith_runtime::{run_admitted, AdmittedRun, AgentSpec, LoopMode};
use bytes::Bytes;
use http_body_util::{BodyExt, Full, Limited};
use hyper::body::Incoming;
use hyper::{header, Method, Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::run::resolve_model;
use super::service::{AdmitError, ServiceInner, MAX_BODY};

pub(crate) async fn handle(
    inner: Arc<ServiceInner>,
    req: Request<Incoming>,
) -> Response<Full<Bytes>> {
    if inner.shutting_down() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "shutting down");
    }
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let headers = req.headers().clone();
    let health = method == Method::GET && path == "/health";
    if !health && !inner.authorized(&headers) {
        let _ = read_body(req).await;
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    match (method, path.as_str()) {
        (Method::GET, "/health") => {
            let _ = read_body(req).await;
            json(StatusCode::OK, &inner.health())
        }
        (Method::POST, "/v1/runs") => {
            let bytes = match read_body(req).await {
                Ok(bytes) => bytes,
                Err(()) => {
                    return error(StatusCode::BAD_REQUEST, "request body is larger than 1 MiB")
                }
            };
            create_run(inner, &headers, &bytes).await
        }
        (Method::GET, path) => {
            let _ = read_body(req).await;
            match run_id(path, "") {
                Some(id) => match inner.view(id) {
                    Some(view) => json(StatusCode::OK, &view),
                    None => error(StatusCode::NOT_FOUND, "not found"),
                },
                None => error(StatusCode::NOT_FOUND, "not found"),
            }
        }
        (Method::POST, path) => {
            if read_body(req).await.is_err() {
                return error(StatusCode::BAD_REQUEST, "request body is larger than 1 MiB");
            }
            match run_id(path, "/cancel") {
                Some(id) => {
                    if inner.cancel_run(id) {
                        json(StatusCode::ACCEPTED, &serde_json::json!({}))
                    } else {
                        error(StatusCode::NOT_FOUND, "not found")
                    }
                }
                None => error(StatusCode::NOT_FOUND, "not found"),
            }
        }
        _ => error(StatusCode::NOT_FOUND, "not found"),
    }
}

fn run_id(path: &str, suffix: &str) -> Option<agentsmith_runtime::RunId> {
    let rest = path.strip_prefix("/v1/runs/")?;
    let id = if suffix.is_empty() {
        rest
    } else {
        rest.strip_suffix(suffix)?
    };
    if id.is_empty() || id.contains('/') {
        return None;
    }
    id.parse().ok()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunRequest {
    task: String,
    #[serde(default)]
    mode: Option<LoopMode>,
    #[serde(default)]
    spec_path: Option<String>,
    #[serde(default)]
    workdir: Option<String>,
}

async fn create_run(
    inner: Arc<ServiceInner>,
    headers: &header::HeaderMap,
    bytes: &[u8],
) -> Response<Full<Bytes>> {
    if inner.shutting_down() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "shutting down");
    }
    let value: serde_json::Value = match serde_json::from_slice::<serde_json::Value>(bytes) {
        Ok(value) if value.is_object() => value,
        _ => return error(StatusCode::BAD_REQUEST, "body must be a JSON object"),
    };
    let body: RunRequest = match serde_json::from_value(value.clone()) {
        Ok(body) => body,
        Err(err) => return error(StatusCode::BAD_REQUEST, &err.to_string()),
    };
    let loaded = match body.spec_path.as_deref() {
        Some(path) => match agentsmith_runtime::load_spec(Path::new(path)) {
            Ok(spec) => Some(spec),
            Err(err) => return error(StatusCode::BAD_REQUEST, &err.to_string()),
        },
        None => None,
    };
    let mode = match agentsmith_runtime::resolve_mode(body.mode, loaded.as_ref()) {
        Ok(mode) => mode,
        Err(err) => return error(StatusCode::BAD_REQUEST, &err.to_string()),
    };
    let spec = loaded.unwrap_or_else(AgentSpec::react_default);
    let model = match resolve_model(&spec, &inner.config.llm.model) {
        Ok(model) => model,
        Err(err) => return error(StatusCode::BAD_REQUEST, &err.to_string()),
    };
    if inner.config.llm.api_key.is_empty() {
        return error(StatusCode::BAD_REQUEST, "llm.api_key is required to run");
    }
    let workdir_path = match body.workdir.as_deref() {
        Some(path) => Path::new(path).to_path_buf(),
        None => inner.config.harness.workdir.clone(),
    };
    let workdir = match workdir_path.canonicalize() {
        Ok(path) => path,
        Err(_) => {
            return error(
                StatusCode::BAD_REQUEST,
                &format!("workdir {} cannot be canonicalized", workdir_path.display()),
            )
        }
    };
    let hash = body_hash(&value);
    let key = idempotency_of(headers);
    let mcp_down = inner.spec_hits_down(&spec);
    let run_cancel = inner.cancel.child_token();
    let admitted = match inner.admit(
        key.as_deref(),
        &hash,
        &workdir,
        mode,
        run_cancel.clone(),
        mcp_down,
    ) {
        Ok(admitted) => admitted,
        Err(AdmitError::Overlap) => {
            return error(StatusCode::CONFLICT, "workdir overlaps an active run")
        }
        Err(AdmitError::Full) => return error(StatusCode::TOO_MANY_REQUESTS, "max_runs reached"),
        Err(AdmitError::Conflict) => {
            return error(
                StatusCode::CONFLICT,
                "idempotency key reused with a different body",
            )
        }
    };
    if admitted.replay {
        return json(
            StatusCode::OK,
            &Created {
                run_id: admitted.run_id.to_string(),
            },
        );
    }
    if mcp_down {
        inner
            .record_mcp_down(admitted.run_id, mode, &spec.name)
            .await;
        return json(
            StatusCode::CREATED,
            &Created {
                run_id: admitted.run_id.to_string(),
            },
        );
    }
    inner
        .events
        .started(admitted.run_id, inner.root_meta(mode, &spec.name))
        .await;
    let run = AdmittedRun {
        run_id: admitted.run_id,
        task: body.task,
        spec,
        config: inner.config.clone(),
        model,
        workdir,
        llm: Arc::clone(&inner.llm),
        cancel: run_cancel,
        mode,
        registry: inner.registry.clone(),
        sessions: inner.sessions.clone(),
        events: Arc::clone(&inner.events),
        max_output_tokens: inner.max_output_tokens,
        limits: inner.limits.clone(),
        spawn_hub: Some(Arc::clone(&inner.hub)),
    };
    let fail_inner = Arc::clone(&inner);
    tokio::spawn(async move {
        let run_id = run.run_id;
        let mode = run.mode;
        if let Err(err) = run_admitted(run).await {
            tracing::warn!("{}", redact(&err.to_string()));
            fail_inner.fail_unfinished(run_id, mode).await;
        }
    });
    json(
        StatusCode::CREATED,
        &Created {
            run_id: admitted.run_id.to_string(),
        },
    )
}

fn idempotency_of(headers: &header::HeaderMap) -> Option<String> {
    let value = headers.get("idempotency-key")?.to_str().ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

pub(crate) fn body_hash(value: &serde_json::Value) -> String {
    // `serde_json::Map` is a `BTreeMap`, so this is compact JSON with sorted keys.
    let canonical = serde_json::to_string(value).unwrap_or_default();
    Sha256::digest(canonical.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

async fn read_body(req: Request<Incoming>) -> Result<Bytes, ()> {
    let limited = Limited::new(req.into_body(), MAX_BODY);
    match limited.collect().await {
        Ok(collected) => Ok(collected.to_bytes()),
        Err(_) => Err(()),
    }
}

#[derive(Serialize)]
struct Created {
    run_id: String,
}

fn error(status: StatusCode, message: &str) -> Response<Full<Bytes>> {
    json(status, &serde_json::json!({ "error": redact(message) }))
}

fn json(status: StatusCode, body: &impl Serialize) -> Response<Full<Bytes>> {
    let bytes = serde_json::to_vec(body).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CONTENT_LENGTH, bytes.len())
        .header(header::CONNECTION, "close")
        .body(Full::new(Bytes::from(bytes)))
        .unwrap_or_else(|_| {
            Response::new(Full::new(Bytes::from_static(b"{\"error\":\"response\"}")))
        })
}
