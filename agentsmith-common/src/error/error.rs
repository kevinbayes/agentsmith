use serde::Serialize;

pub type SystemResult<T> = Result<T, SystemError>;

#[derive(Clone, Debug, Serialize, strum_macros::AsRefStr)]
#[serde(tag = "type", content = "data")]
pub enum SystemError {
    //General
    NotImplemented,
    LoginFail,
    JwksError,

    EmbeddingError { id: u8, code: u16 },

    //Database
    IncrementNumberFailed {id: u8} ,

    //Authentication and Authorisation
    AuthFailed,
    AuthError { id: u8, code: u16 },

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
impl core::fmt::Display for SystemError {
    fn fmt(
        &self,
        fmt: &mut core::fmt::Formatter,
    ) -> core::result::Result<(), core::fmt::Error> {
        write!(fmt, "{self:?}")
    }
}

impl std::error::Error for SystemError {}
// endregion: --- Error Boilerplate

#[allow(non_camel_case_types)]
pub enum ClientError {
    LOGIN_FAIL,
    NO_AUTH,
    NOT_FOUND,
    UNSUPPORTED,
    INVALID_PARAMS,
    SERVICE_ERROR,
}

/*
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Debug, Serialize, strum_macros::AsRefStr)]
#[serde(tag = "type", content = "data")]
pub enum Error {
    NotImplemented,
    LoginFail,
    JwksError,



    // -- Auth errors.
    AuthFailed,
    InvalidToken,
    AuthFailNoAuthTokenCookie,
    AuthFailTokenWrongFormat,
    AuthFailCtxNotInRequestExt,

    // -- Account errors.
    ProfileMeNotFound,
    GetProfileError,
    GetAccountError,

    // -- Model errors.
    TicketDeleteFailIdNotFound { id: u64 },
    GetPersonError,
    PersonNotFound,

    // -- Fallback.
    IncrementNumberFailed{ id: u8 },
    TransactionError{ id: u8 },
    ProfileNotCreated{ id: u8 },

    // Create conflict
    ConflictUploadError { id: u8, code: u16 },

    //Agent
    AgentError { id: u8, code: u16 },

}

// region:    --- Error Boilerplate
impl core::fmt::Display for Error {
    fn fmt(
        &self,
        fmt: &mut core::fmt::Formatter,
    ) -> core::result::Result<(), core::fmt::Error> {
        write!(fmt, "{self:?}")
    }
}

impl std::error::Error for Error {}
// endregion: --- Error Boilerplate

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        println!("->> {:<12} - {self:?}", "INTO_RES");

        // Create a placeholder Axum reponse.
        let mut response = StatusCode::INTERNAL_SERVER_ERROR.into_response();

        // Insert the Error into the reponse.
        response.extensions_mut().insert(self);

        response
    }
}

impl Error {
    pub fn client_status_and_error(&self) -> (StatusCode, ClientError) {
        #[allow(unreachable_patterns)]
        match self {
            Self::LoginFail => (StatusCode::FORBIDDEN, ClientError::LOGIN_FAIL),

            // -- Auth.
            Self::AuthFailNoAuthTokenCookie
            | Self::AuthFailTokenWrongFormat
            | Self::AuthFailCtxNotInRequestExt => {
                (StatusCode::FORBIDDEN, ClientError::NO_AUTH)
            }


            // -- Auth.
            Self::ProfileMeNotFound => {
                (StatusCode::NOT_FOUND, ClientError::NOT_FOUND)
            }

            // -- Model.
            Self::TicketDeleteFailIdNotFound { .. } => {
                (StatusCode::BAD_REQUEST, ClientError::INVALID_PARAMS)
            }

            // -- Fallback.
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                ClientError::SERVICE_ERROR,
            ),
        }
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
 */