use std::collections::HashMap;
use std::env;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use jsonwebtoken::{encode, Header, EncodingKey, TokenData, decode, DecodingKey, Validation, Algorithm, decode_header};
use jsonwebtoken::errors::ErrorKind;
use serde::{Serialize, Deserialize};
use crate::config::config::Config;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmithClaims {
    pub iss: String,
    pub aud: String,
    pub sub: String,
    pub iat: Option<u64>,
    pub exp: u64,
    pub tnt: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GcpIdentityPlatformClaims {
    pub iss: String,
    pub aud: String,
    pub sub: String,
    pub iat: Option<u64>,
    pub exp: u64,
    pub user_id: String,
}

pub fn decode_jwt(config: Config, token: &str) -> Result<TokenData<SmithClaims>, jsonwebtoken::errors::Error> {
    let secret_key = env::var("SHARED_KEY").unwrap_or_else(|_| config.config.security.jwt.secret);
    let mut validation = Validation::new(Algorithm::HS256);
    validation.set_issuer(&["agentsmith"]);
    validation.set_audience(&["agentsmith"]);
    decode::<SmithClaims>(token, &DecodingKey::from_secret(secret_key.as_bytes()), &validation)
}

pub fn generate_jwt(
    config: Config,
    aud: &str,
    sub: &str,
    tnt: &str) -> Result<String, jsonwebtoken::errors::Error> {

    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_else(|_| Duration::from_secs(0)).as_secs();

    let claims = SmithClaims {
        iss: sub.to_owned(),
        aud: aud.to_owned(),
        sub: sub.to_owned(),
        tnt: tnt.to_owned(),
        iat: Some(now),
        exp: now + 7200,
    };


    let secret_key = env::var("SHARED_SECRET").unwrap_or_else(|_| config.config.security.jwt.secret);
    encode(&Header::default(), &claims, &EncodingKey::from_secret(secret_key.as_bytes()))
}




pub fn decode_google_jwt<T: ToString>(certs: HashMap<String, String>, raw_token: &str, issuer: &[T], audience: &[T], validate_exp: bool) -> Result<TokenData<GcpIdentityPlatformClaims>, jsonwebtoken::errors::Error> {
    let mut validation = Validation::new(Algorithm::RS256);
    validation.set_audience(audience);
    validation.set_issuer(issuer);
    validation.set_required_spec_claims(&["iss", "aud"]);
    validation.validate_exp = validate_exp;

    let token = raw_token.replace("Bearer ", "");

    let header = match decode_header(token.clone().as_str()) {
        Ok(_header) => _header,
        Err(e) => return Err(e),
    };

    let kid = match header.kid.ok_or(jsonwebtoken::errors::Error::from(ErrorKind::InvalidAlgorithmName)) {
        Ok(_kid) => _kid,
        Err(e) => return Err(e)
    };

    let cert = match certs.get(&kid).ok_or(jsonwebtoken::errors::Error::from(ErrorKind::InvalidToken)) {
        Ok(_cert) => _cert,
        Err(e) => return Err(e)
    };

    decode::<GcpIdentityPlatformClaims>(token.as_str(), &DecodingKey::from_rsa_pem(cert.as_bytes()).unwrap(), &validation)
}



#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_decode_google_jwt() {

        let certs: HashMap<String, String> = HashMap::from([
            (String::from("a238dd04cbaa580b304c881e1c008ec28fbbaddc"), String::from("-----BEGIN CERTIFICATE-----\nMIIDHTCCAgWgAwIBAgIJAObMlT0V930JMA0GCSqGSIb3DQEBBQUAMDExLzAtBgNV\nBAMMJnNlY3VyZXRva2VuLnN5c3RlbS5nc2VydmljZWFjY291bnQuY29tMB4XDTI0\nMDQyMjA3MzIyMFoXDTI0MDUwODE5NDcyMFowMTEvMC0GA1UEAwwmc2VjdXJldG9r\nZW4uc3lzdGVtLmdzZXJ2aWNlYWNjb3VudC5jb20wggEiMA0GCSqGSIb3DQEBAQUA\nA4IBDwAwggEKAoIBAQDdtOxC3HFjVHEZkenoiDQKMrJOzUbzlS7j0qOWfmY4XMst\nxS9V/9ToiBg+5l2YozafcXaWppNIDaMLax3Zo0/Pi4qduB2gT6pBgawBuuu0e7jo\nB/D6+cokfBLzzSK1xBcFNhQRHbDPLtfy6iaYBlBl7Mxpe9t2hX7DOlR1LHpZ/Zhu\nP1j10V3bJZIg4Ot5scyVz3WExIXJGQyc3qTMFumOqKJPHFyPIdZXytaehtlsgmdB\nrHvkjVU9Hde4pUI7nOHExZvyeKzn8WvGziIiAaBk7p9t4Vsrjpsr8XB6iX95e6z/\nr9cki1/ndvGPQF1wGFXGQsWRwsfBZjQC9lf2BsYzAgMBAAGjODA2MAwGA1UdEwEB\n/wQCMAAwDgYDVR0PAQH/BAQDAgeAMBYGA1UdJQEB/wQMMAoGCCsGAQUFBwMCMA0G\nCSqGSIb3DQEBBQUAA4IBAQBU4xTBbynw/6qiLRIkc7veLNs08IUVOiaedSkGptXf\niQla6wmzRJOi9CXgMQQuJnJfQGdoZ8sUJLYlPj2SrGKZimkIW8nsu4YYyGzKtyuS\n9fslFjHPrLzhJTRs0w3hnR+1EwXDmSHZDtjSyEA0KBMxWi1m5SOtZEboq4xsN8BC\ndGvkawDGD0NtRWq0bRhN1p+m1hdIWwpxFdP6nKPZRMmxJ+H54UZXZjNYZcpxSAQj\n1EDqb/FWavYNW4Z34NZ1avQeO0PfwzTbe1TfO8iy0qURtmD8zYnp6mAn/ANtG7E2\nkkl2mWp7ONyCDYV4rmCHgcye7UnwDio3m7Ns6W15LDmV\n-----END CERTIFICATE-----\n")),
            (String::from("2d9b4e69e32b7615d4cd7cafb8fc9b3f81a41ac0"), String::from("-----BEGIN CERTIFICATE-----\nMIIDHTCCAgWgAwIBAgIJANqx+xhL/lXpMA0GCSqGSIb3DQEBBQUAMDExLzAtBgNV\nBAMMJnNlY3VyZXRva2VuLnN5c3RlbS5nc2VydmljZWFjY291bnQuY29tMB4XDTI0\nMDQxNDA3MzIxOVoXDTI0MDQzMDE5NDcxOVowMTEvMC0GA1UEAwwmc2VjdXJldG9r\nZW4uc3lzdGVtLmdzZXJ2aWNlYWNjb3VudC5jb20wggEiMA0GCSqGSIb3DQEBAQUA\nA4IBDwAwggEKAoIBAQDOufB+hyxN+t1k55+k0qO/I903188BC2SZEnl3fwq1B/Fl\n0vbGwNKsYBAOZku/1D8dNjOnSH7X47ShHGuPSrvWGmbEdYZ//40jsPjkQW2fBMEr\nPfBxrAZDhIRQtbh9DWk7gi14Kghz0B7f2d4iQwZjtNQkyoJvlPfxfktdYwYFzyBt\nIw85/CqH39QEFEVGR45ck40htg+cYjMN3bXejhOXiqzkS4EM+Me/6yFEYqb5yp01\n+vLE7/lyEOKNZHOzq/8t6srWEayo3L7eaCiCjvRUOx8xHavJ6DH9jDBqsN9H6lby\nVMJv6/cPXDE7LKBgKrcEd/nghAPBZDHjDBCLOsg5AgMBAAGjODA2MAwGA1UdEwEB\n/wQCMAAwDgYDVR0PAQH/BAQDAgeAMBYGA1UdJQEB/wQMMAoGCCsGAQUFBwMCMA0G\nCSqGSIb3DQEBBQUAA4IBAQA1e+ZZAwkD0TLXnn9SiPrHTdXcNaRZbGIbJboGGi42\nfqSGKErVqcUFDCol1TsLGoN5YK2hTiytTyN/DI3Ot+aB3LYqa61S2Nn2OeaRDyHf\nu0IxpbnhXX9NBoWG2Q1Dbf8v9hUJjwQxkas1LQ9PRZBGFGXHhoPDviNDvYe5QUFU\nqz8wyheErUUZVdQZ+CGDgWWSvljBxg97eWEDSZ1TInTecg0suNLTRD8RQnDAL8RN\nHFzFZ8HYS6TzeVNLmynhDdtO02isZVyL2YqeTYgnaZEyw90t1czoWTX4Um6vUIUG\neUuZElsh7GyBtB0Mlymby+ES08Sjg/W8Em6Z0KpIorXz\n-----END CERTIFICATE-----\n")),
        ]);
        let token = "Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6ImEyMzhkZDA0Y2JhYTU4MGIzMDRjODgxZTFjMDA4ZWMyOGZiYmFkZGMiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vZGgtcGF0aWVudHBsdXMtdGVzdCIsImF1ZCI6ImRoLXBhdGllbnRwbHVzLXRlc3QiLCJhdXRoX3RpbWUiOjE3MTQxMzY3MTYsInVzZXJfaWQiOiJ4VjZhSVk2eUlHYk1wazhyVW1SMjh0U1Q1RFkyIiwic3ViIjoieFY2YUlZNnlJR2JNcGs4clVtUjI4dFNUNURZMiIsImlhdCI6MTcxNDEzNjcxNiwiZXhwIjoxNzE0MTQwMzE2LCJlbWFpbCI6ImtldmluQGRlY29kZWRoZWFsdGguY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImZpcmViYXNlIjp7ImlkZW50aXRpZXMiOnsiZW1haWwiOlsia2V2aW5AZGVjb2RlZGhlYWx0aC5jb20iXX0sInNpZ25faW5fcHJvdmlkZXIiOiJwYXNzd29yZCIsInRlbmFudCI6InBhdGllbnRwbHVzLTV5anFwIn19.BwOkBPFRMlkIyRI9RSZCnbNDvD_IkcIsvSf9hKr10hWwr6mKJYghTjwOerAL7Fn54Z-Sgig1gATjsdijNtjVvDydNSfe5Eee17O4-uwl0npujTsNy0Wxmmft0DAfbHGMiobnhM_A5r7FJZ3JsDZdUML_t3hagVCnS-uARKZcOVopdt6UP8xv45vISrzqHfQ-5Yg3o98w32MvMi3O2h32MFi10urZ6E2vNwRuQ8-L7gYlY1t4Nnl7KRwIONg3MBUGIxCQeoX4-ipyCXdo0WFzuAsAcSOuy5Y8XA_U4asv0JjYYoeAAlUg0qtNEDnSnG_zMf5WX70xYDqUY7vUKvhSnw";
        let audience = "agentsmith";
        let issuer = "https://securetoken.google.com/agentsmith";

        let result = decode_google_jwt(certs, token, &[issuer], &[audience], false);
        result.unwrap();
    }
}