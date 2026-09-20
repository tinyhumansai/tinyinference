//! Host-neutral OAuth authorization-code flow with PKCE.

use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::RngExt as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const PENDING_TTL_SECS: u64 = 600;
const HTTP_TIMEOUT_SECS: u64 = 20;

/// OpenAI Codex public OAuth client identifier.
pub const OPENAI_CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
/// OpenAI authorization endpoint.
pub const OPENAI_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
/// OpenAI token endpoint.
pub const OPENAI_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

/// Builds the public OpenAI Codex OAuth configuration.
#[must_use]
pub fn openai_codex_config(redirect_uri: impl Into<String>) -> OAuthConfig {
    OAuthConfig {
        client_id: OPENAI_CODEX_CLIENT_ID.to_string(),
        client_secret: None,
        authorize_url: OPENAI_AUTHORIZE_URL.to_string(),
        token_url: OPENAI_TOKEN_URL.to_string(),
        redirect_uri: redirect_uri.into(),
        scopes: ["openid", "profile", "email", "offline_access"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        extra_authorize_params: vec![("access_type".to_string(), "offline".to_string())],
        state_equals_verifier: false,
        pending_filename: "openai-oauth-pending.json".to_string(),
    }
}

/// OAuth client and endpoint configuration.
#[derive(Clone, PartialEq, Eq)]
pub struct OAuthConfig {
    /// OAuth client identifier.
    pub client_id: String,
    /// Optional client secret.
    pub client_secret: Option<String>,
    /// Authorization endpoint.
    pub authorize_url: String,
    /// Token endpoint.
    pub token_url: String,
    /// Redirect URI registered by the client.
    pub redirect_uri: String,
    /// Requested scopes.
    pub scopes: Vec<String>,
    /// Additional provider-specific authorization query parameters.
    pub extra_authorize_params: Vec<(String, String)>,
    /// Use the PKCE verifier itself as state when required by a provider.
    pub state_equals_verifier: bool,
    /// Filename used for the short-lived pending session.
    pub pending_filename: String,
}

impl std::fmt::Debug for OAuthConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OAuthConfig")
            .field("client_id", &self.client_id)
            .field(
                "client_secret",
                &self.client_secret.as_ref().map(|_| "[REDACTED]"),
            )
            .field("authorize_url", &self.authorize_url)
            .field("token_url", &self.token_url)
            .field("redirect_uri", &self.redirect_uri)
            .field("scopes", &self.scopes)
            .field("extra_authorize_params", &self.extra_authorize_params)
            .field("state_equals_verifier", &self.state_equals_verifier)
            .field("pending_filename", &self.pending_filename)
            .finish()
    }
}

/// Persistable OAuth token set owned by the host's credential store.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OAuthTokenSet {
    /// Access token.
    pub access_token: String,
    /// Optional refresh token.
    pub refresh_token: Option<String>,
    /// Optional ID token.
    pub id_token: Option<String>,
    /// Lifetime reported by the provider.
    pub expires_in: u64,
    /// Unix timestamp when the token was issued.
    pub issued_at: u64,
}

impl std::fmt::Debug for OAuthTokenSet {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OAuthTokenSet")
            .field("access_token", &"[REDACTED]")
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "[REDACTED]"),
            )
            .field("id_token", &self.id_token.as_ref().map(|_| "[REDACTED]"))
            .field("expires_in", &self.expires_in)
            .field("issued_at", &self.issued_at)
            .finish()
    }
}

/// Result returned when an OAuth flow begins.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OAuthStart {
    /// URL the user should open.
    pub auth_url: String,
    /// CSRF state value expected on callback.
    pub state: String,
    /// Redirect URI expected on callback.
    pub redirect_uri: String,
}

/// Tokens imported from the OpenAI Codex CLI credential file.
#[derive(Clone, PartialEq, Eq)]
pub struct ImportedOpenAiCredentials {
    /// Normalized OAuth tokens.
    pub token: OAuthTokenSet,
    /// ChatGPT account identifier, when present in the file or access token.
    pub account_id: Option<String>,
    /// Access-token expiry from its JWT payload.
    pub expires_at_unix: Option<i64>,
}

impl std::fmt::Debug for ImportedOpenAiCredentials {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ImportedOpenAiCredentials")
            .field("token", &self.token)
            .field("account_id", &self.account_id)
            .field("expires_at_unix", &self.expires_at_unix)
            .finish()
    }
}

/// Parses an OpenAI Codex CLI `auth.json` payload.
pub fn parse_openai_codex_auth_json(bytes: &[u8]) -> Result<ImportedOpenAiCredentials, String> {
    #[derive(Deserialize)]
    struct AuthFile {
        tokens: Option<AuthTokens>,
    }
    #[derive(Deserialize)]
    struct AuthTokens {
        access_token: Option<String>,
        refresh_token: Option<String>,
        id_token: Option<String>,
        account_id: Option<String>,
    }

    let parsed: AuthFile = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    let tokens = parsed
        .tokens
        .ok_or_else(|| "Codex CLI auth has no tokens".to_string())?;
    let access_token = tokens.access_token.unwrap_or_default().trim().to_string();
    if access_token.is_empty() {
        return Err("Codex CLI auth has no access token".to_string());
    }
    let refresh_token = normalize_optional(tokens.refresh_token);
    let id_token = normalize_optional(tokens.id_token);
    let account_id = normalize_optional(tokens.account_id)
        .or_else(|| openai_account_id_from_access_token(&access_token));
    let expires_at_unix = openai_access_token_expiry(&access_token);
    Ok(ImportedOpenAiCredentials {
        token: OAuthTokenSet {
            access_token,
            refresh_token,
            id_token,
            expires_in: 0,
            issued_at: unix_now_secs(),
        },
        account_id,
        expires_at_unix,
    })
}

/// Extracts the ChatGPT account identifier from an OpenAI access-token JWT.
#[must_use]
pub fn openai_account_id_from_access_token(access_token: &str) -> Option<String> {
    let json = decode_jwt_payload(access_token)?;
    json.get("https://api.openai.com/auth")
        .and_then(|value| value.get("chatgpt_account_id"))
        .or_else(|| json.get("chatgpt_account_id"))
        .or_else(|| json.get("sub"))
        .or_else(|| json.get("account_id"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn openai_access_token_expiry(access_token: &str) -> Option<i64> {
    decode_jwt_payload(access_token)?
        .get("exp")
        .and_then(|value| value.as_i64().or_else(|| value.as_str()?.parse().ok()))
}

fn decode_jwt_payload(access_token: &str) -> Option<serde_json::Value> {
    let payload = access_token.split('.').nth(1)?;
    let padded = match payload.len() % 4 {
        0 => payload.to_string(),
        remainder => format!("{}{}", payload, "=".repeat(4 - remainder)),
    };
    let bytes = base64::engine::general_purpose::URL_SAFE
        .decode(padded.as_bytes())
        .or_else(|_| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(payload.as_bytes()))
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn normalize_optional(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let value = value.trim().to_string();
        (!value.is_empty()).then_some(value)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unsigned_jwt(payload: serde_json::Value) -> String {
        let header = URL_SAFE_NO_PAD.encode(r#"{"alg":"none"}"#);
        let payload = URL_SAFE_NO_PAD.encode(payload.to_string());
        format!("{header}.{payload}.")
    }

    #[test]
    fn codex_config_contains_the_public_client_contract() {
        let config = openai_codex_config("http://127.0.0.1:1455/auth/callback");
        assert_eq!(config.client_id, OPENAI_CODEX_CLIENT_ID);
        assert_eq!(config.authorize_url, OPENAI_AUTHORIZE_URL);
        assert_eq!(config.token_url, OPENAI_TOKEN_URL);
        assert!(config.scopes.iter().any(|scope| scope == "offline_access"));
    }

    #[test]
    fn codex_cli_parser_normalizes_tokens_and_reads_jwt_metadata() {
        let access_token = unsigned_jwt(serde_json::json!({
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"},
            "exp": 2_000_000_000_i64,
        }));
        let bytes = serde_json::to_vec(&serde_json::json!({
            "tokens": {
                "access_token": access_token,
                "refresh_token": " refresh ",
                "id_token": " id "
            }
        }))
        .expect("fixture");
        let imported = parse_openai_codex_auth_json(&bytes).expect("parse");
        assert_eq!(imported.account_id.as_deref(), Some("acct_123"));
        assert_eq!(imported.expires_at_unix, Some(2_000_000_000));
        assert_eq!(imported.token.refresh_token.as_deref(), Some("refresh"));
        assert_eq!(imported.token.id_token.as_deref(), Some("id"));
    }

    #[test]
    fn codex_cli_parser_rejects_missing_access_token() {
        let error = parse_openai_codex_auth_json(br#"{"tokens":{}}"#).unwrap_err();
        assert!(error.contains("access token"));
    }

    #[test]
    fn debug_output_redacts_oauth_credentials() {
        let token = OAuthTokenSet {
            access_token: "access-secret".to_string(),
            refresh_token: Some("refresh-secret".to_string()),
            id_token: Some("id-secret".to_string()),
            expires_in: 3600,
            issued_at: 1,
        };
        let debug = format!("{token:?}");
        assert!(!debug.contains("access-secret"));
        assert!(!debug.contains("refresh-secret"));
        assert!(!debug.contains("id-secret"));
        assert!(debug.contains("[REDACTED]"));
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct PendingOAuth {
    state: String,
    verifier: String,
    redirect_uri: String,
    created_at: u64,
}

/// File-backed PKCE flow. Credential persistence remains a host concern.
#[derive(Clone, Debug)]
pub struct OAuthFlow {
    config: OAuthConfig,
    pending_path: PathBuf,
}

impl OAuthFlow {
    /// Creates a flow whose short-lived pending state is rooted in `state_dir`.
    pub fn new(config: OAuthConfig, state_dir: impl AsRef<Path>) -> Self {
        let pending_path = state_dir.as_ref().join(&config.pending_filename);
        Self {
            config,
            pending_path,
        }
    }

    /// Creates PKCE state, persists it, and returns the authorization URL.
    pub fn start(&self) -> Result<OAuthStart, String> {
        let (verifier, challenge) = generate_pkce();
        let state = if self.config.state_equals_verifier {
            verifier.clone()
        } else {
            random_state()
        };
        self.write_pending(&PendingOAuth {
            state: state.clone(),
            verifier,
            redirect_uri: self.config.redirect_uri.clone(),
            created_at: unix_now_secs(),
        })?;
        Ok(OAuthStart {
            auth_url: authorization_url(&self.config, &state, &challenge)?,
            state,
            redirect_uri: self.config.redirect_uri.clone(),
        })
    }

    /// Validates a callback and exchanges its authorization code for tokens.
    pub async fn complete(&self, callback: &str) -> Result<OAuthTokenSet, String> {
        let pending = self
            .read_pending()?
            .ok_or_else(|| "no pending OAuth session; start the OAuth flow first".to_string())?;
        let (code, returned_state) = parse_callback_input(callback)?;
        if returned_state != pending.state {
            self.clear_pending();
            return Err("OAuth state mismatch — try connecting again".to_string());
        }
        let token = exchange_authorization_code(
            &self.config,
            &code,
            &pending.verifier,
            &pending.redirect_uri,
        )
        .await?;
        self.clear_pending();
        Ok(token)
    }

    /// Removes any pending authorization state.
    pub fn clear_pending(&self) {
        let _ = std::fs::remove_file(&self.pending_path);
    }

    fn write_pending(&self, pending: &PendingOAuth) -> Result<(), String> {
        if let Some(parent) = self.pending_path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let json = serde_json::to_vec_pretty(pending).map_err(|error| error.to_string())?;
        std::fs::write(&self.pending_path, json).map_err(|error| error.to_string())
    }

    fn read_pending(&self) -> Result<Option<PendingOAuth>, String> {
        let bytes = match std::fs::read(&self.pending_path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.to_string()),
        };
        if bytes.is_empty() {
            return Ok(None);
        }
        let pending: PendingOAuth =
            serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        if unix_now_secs().saturating_sub(pending.created_at) > PENDING_TTL_SECS {
            self.clear_pending();
            return Ok(None);
        }
        Ok(Some(pending))
    }
}

/// Parses a redirect URL or raw query into its code and state values.
pub fn parse_callback_input(input: &str) -> Result<(String, String), String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("callback URL is required".to_string());
    }
    let query = if let Ok(parsed) = url::Url::parse(trimmed) {
        parsed.query().unwrap_or_default().to_owned()
    } else if trimmed.contains('=') {
        trimmed.to_owned()
    } else {
        return Err("invalid callback URL".to_string());
    };
    let mut code = None;
    let mut state = None;
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        match key.as_ref() {
            "code" if !value.is_empty() => code = Some(value.into_owned()),
            "state" if !value.is_empty() => state = Some(value.into_owned()),
            _ => {}
        }
    }
    Ok((
        code.ok_or_else(|| "callback URL missing code parameter".to_string())?,
        state.ok_or_else(|| "callback URL missing state parameter".to_string())?,
    ))
}

/// Builds a provider authorization URL.
pub fn authorization_url(
    config: &OAuthConfig,
    state: &str,
    code_challenge: &str,
) -> Result<String, String> {
    let mut url = url::Url::parse(&config.authorize_url).map_err(|error| error.to_string())?;
    {
        let mut query = url.query_pairs_mut();
        query
            .append_pair("client_id", &config.client_id)
            .append_pair("response_type", "code")
            .append_pair("redirect_uri", &config.redirect_uri)
            .append_pair("scope", &config.scopes.join(" "))
            .append_pair("state", state)
            .append_pair("code_challenge", code_challenge)
            .append_pair("code_challenge_method", "S256");
        for (key, value) in &config.extra_authorize_params {
            query.append_pair(key, value);
        }
    }
    Ok(url.into())
}

/// Exchanges an authorization code using RFC 7636 PKCE.
pub async fn exchange_authorization_code(
    config: &OAuthConfig,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<OAuthTokenSet, String> {
    let mut params = vec![
        ("grant_type", "authorization_code"),
        ("code", code),
        ("redirect_uri", redirect_uri),
        ("code_verifier", verifier),
        ("client_id", config.client_id.as_str()),
    ];
    if let Some(secret) = config.client_secret.as_deref() {
        params.push(("client_secret", secret));
    }
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(HTTP_TIMEOUT_SECS))
        .build()
        .map_err(|error| error.to_string())?
        .post(&config.token_url)
        .header("Accept", "application/json")
        .form(&params)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("HTTP {status}: {body}"));
    }
    let raw: RawTokenResponse = response.json().await.map_err(|error| error.to_string())?;
    Ok(OAuthTokenSet {
        access_token: raw.access_token,
        refresh_token: raw.refresh_token,
        id_token: raw.id_token,
        expires_in: raw.expires_in,
        issued_at: unix_now_secs(),
    })
}

/// Exchanges a refresh token for a current OAuth token set.
pub async fn refresh_access_token(
    config: &OAuthConfig,
    refresh_token: &str,
) -> Result<OAuthTokenSet, String> {
    let mut params = vec![
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", config.client_id.as_str()),
    ];
    if let Some(secret) = config.client_secret.as_deref() {
        params.push(("client_secret", secret));
    }
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(HTTP_TIMEOUT_SECS))
        .build()
        .map_err(|error| error.to_string())?
        .post(&config.token_url)
        .header("Accept", "application/json")
        .form(&params)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("HTTP {status}: {body}"));
    }
    let raw: RawTokenResponse = response.json().await.map_err(|error| error.to_string())?;
    Ok(OAuthTokenSet {
        access_token: raw.access_token,
        refresh_token: raw
            .refresh_token
            .or_else(|| Some(refresh_token.to_string())),
        id_token: raw.id_token,
        expires_in: raw.expires_in,
        issued_at: unix_now_secs(),
    })
}

#[derive(Deserialize)]
struct RawTokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    id_token: Option<String>,
    expires_in: u64,
}

fn generate_pkce() -> (String, String) {
    let bytes: [u8; 64] = rand::rng().random();
    let verifier = URL_SAFE_NO_PAD.encode(bytes);
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    (verifier, challenge)
}

fn random_state() -> String {
    URL_SAFE_NO_PAD.encode(rand::rng().random::<[u8; 16]>())
}

fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
