/// A new UUIDv7. Run ids use this.
pub fn new_id() -> uuid::Uuid {
    uuid::Uuid::now_v7()
}

/// First 12 hex chars of SHA-256. Config fingerprints and task ids share this.
pub fn sha256_12(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    digest.iter().take(6).fold(String::new(), |mut out, byte| {
        out.push_str(&format!("{byte:02x}"));
        out
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_id_is_uuid_v7() {
        let id = new_id();
        assert_eq!(id.get_version(), Some(uuid::Version::SortRand));
    }
}
