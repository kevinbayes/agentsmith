use std::path::Path;

use agentsmith_core::config;
use agentsmith_core::Error;

pub fn check(config_path: Option<&Path>) -> Result<(), Error> {
    let loaded = config::load(config_path)?;
    println!("{}", loaded.fingerprint()?);
    Ok(())
}
