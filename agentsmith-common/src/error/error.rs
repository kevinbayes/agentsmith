use serde::Serialize;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Debug, Serialize, strum_macros::AsRefStr)]
#[serde(tag = "type", content = "data")]
pub enum Error {
    //General
    NotImplemented,
    LoginFail,
    JwksError,

    //Agent
    AgentFactoryError { id: u8, code: u16 },
    AgentError { id: u8, code: u16 },

    //Tool
    ToolFactoryError { id: u8, code: u16 },
    ToolError { id: u8, code: u16 },

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

#[allow(non_camel_case_types)]
pub enum ClientError {
    LOGIN_FAIL,
    NO_AUTH,
    NOT_FOUND,
    UNSUPPORTED,
    INVALID_PARAMS,
    SERVICE_ERROR,
}