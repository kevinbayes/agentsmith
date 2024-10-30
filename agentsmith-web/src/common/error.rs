use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use crate::common::web::AppJson;

pub type WebResult<T> = core::result::Result<T, WebError>;

#[derive(Clone, Debug, Serialize, strum_macros::AsRefStr)]
#[serde(tag = "type", content = "data")]
pub enum WebError {
    //General
    NotImplemented,
    LoginFail,
    JwksError,

    //Person
    AccountError { id: u8, code: u16 },

    //Person
    PersonError { id: u8, code: u16 },

    //Embedding
    EmbeddingError { id: u8, code: u16 },
    VectorError { id: u8, code: u16 },

    //Database
    IncrementNumberFailed {id: u8} ,
    DatabaseError {id: u8, code: u16} ,

    //Authentication and Authorisation
    AuthFailed,
    AuthError { id: i32, code: i32 },

    //LLM
    LLMFactoryError { id: u8, code: u16 },
    LLMError { id: u8, code: u16 },

    //Agent
    AgentFactoryError { id: u8, code: u16 },
    AgentError { id: u8, code: u16 },

    //Agent
    MemoryFactoryError { id: u8, code: u16 },
    MemoryError { id: u8, code: u16 },

    //Tool
    ToolFactoryError { id: u8, code: u16 },
    ToolError { id: u8, code: u16 },
}

// region:    --- Error Boilerplate
impl core::fmt::Display for WebError {
    fn fmt(
        &self,
        fmt: &mut core::fmt::Formatter,
    ) -> core::result::Result<(), core::fmt::Error> {
        write!(fmt, "{self:?}")
    }
}

impl std::error::Error for WebError {}
// endregion: --- Error Boilerplate

impl IntoResponse for WebError {
    fn into_response(self) -> Response {

        #[derive(Serialize)]
        struct ErrorResponse {
            id: u8,
            code: u16,
            message: String,
        }

        let (status, message, id, code) = match self {
            WebError::EmbeddingError {id,code} => {
                // This error is caused by bad user input so don't log it
                (StatusCode::INTERNAL_SERVER_ERROR, "Embedding error".to_string(), id, code)
            }
            _ => {
                (StatusCode::INTERNAL_SERVER_ERROR, "Unknown error".to_string(), 0, 0)
            }
        };

        (status, AppJson(ErrorResponse { id, code, message })).into_response()
    }
}

#[derive(Debug, strum_macros::AsRefStr)]
#[allow(non_camel_case_types)]
pub enum ClientError {
    LOGIN_FAIL,
    NO_AUTH,
    NOT_FOUND,
    UNSUPPORTED,
    INVALID_PARAMS,
    SERVICE_ERROR,
}


impl WebError {
    pub fn client_status_and_error(&self) -> (StatusCode, ClientError) {
        #[allow(unreachable_patterns)]
        match self {
            Self::LoginFail => (StatusCode::FORBIDDEN, ClientError::LOGIN_FAIL),

            // -- Auth.
            Self::AuthError {id: 0, code: 0}
            | Self::AuthFailed => {
                (StatusCode::FORBIDDEN, ClientError::NO_AUTH)
            }


            // -- Auth.
            Self::AccountError {id: 0, code: 0} => {
                (StatusCode::NOT_FOUND, ClientError::NOT_FOUND)
            }

            // -- Fallback.
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                ClientError::SERVICE_ERROR,
            ),
        }
    }
}
