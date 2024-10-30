use axum::extract::FromRequest;
use axum::response::{ IntoResponse, Response };
use crate::common::error::WebError;

#[derive(FromRequest)]
#[from_request(via(axum::Json), rejection(WebError))]
pub struct AppJson<T>(pub T);

impl<T> IntoResponse for AppJson<T>
where
    axum::Json<T>: IntoResponse,
{
    fn into_response(self) -> Response {
        axum::Json(self.0).into_response()
    }
}
