//! OCI credential models and the signing-key sources behind them.
//!
//! [`OciAuth`] is what a host configures. Resolving it yields either a bearer
//! token (OCI Generative AI API keys) or a signing key source used by the
//! [`OciRequestSigner`](super::signer::OciRequestSigner).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::signer::SigningKey;
use crate::{Error, Result};

/// Environment variable naming the resource-principal version.
const RP_VERSION_ENV: &str = "OCI_RESOURCE_PRINCIPAL_VERSION";
/// Environment variable holding the RPST token (or an absolute path to it).
const RP_RPST_ENV: &str = "OCI_RESOURCE_PRINCIPAL_RPST";
/// Environment variable holding the session private PEM (or an absolute path).
const RP_PRIVATE_PEM_ENV: &str = "OCI_RESOURCE_PRINCIPAL_PRIVATE_PEM";
/// Environment variable holding the optional PEM passphrase (unsupported).
const RP_PASSPHRASE_ENV: &str = "OCI_RESOURCE_PRINCIPAL_PRIVATE_PEM_PASSPHRASE";
/// Environment variable holding the resource-principal region.
const RP_REGION_ENV: &str = "OCI_RESOURCE_PRINCIPAL_REGION";

/// A session token plus the private key whose public half it is bound to.
#[derive(Clone)]
pub struct OciSessionMaterial {
    /// The security token (the `keyId` becomes `ST$<token>`).
    pub token: String,
    /// Unencrypted PEM private key matching the token.
    pub private_key_pem: String,
}

impl std::fmt::Debug for OciSessionMaterial {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("OciSessionMaterial([redacted])")
    }
}

/// Supplies fresh session material on demand (token refresh, rotated files,
/// federation). Called once per signed request, so implementations should be
/// cheap or cache internally.
pub trait OciTokenSource: Send + Sync + std::fmt::Debug {
    /// Returns the current token and key.
    ///
    /// # Errors
    ///
    /// Returns an error when the material cannot be obtained.
    fn fetch(&self) -> Result<OciSessionMaterial>;
}

/// How requests to OCI are authenticated.
#[derive(Clone)]
pub enum OciAuth {
    /// An OCI Generative AI API key sent as `Authorization: Bearer`.
    ApiKey {
        /// The API key.
        bearer: String,
    },
    /// Classic user API-key signing.
    UserPrincipal {
        /// Tenancy OCID.
        tenancy: String,
        /// User OCID.
        user: String,
        /// Public key fingerprint (`aa:bb:...`).
        fingerprint: String,
        /// Unencrypted PEM private key.
        private_key_pem: String,
    },
    /// A session token and its key, fixed at construction.
    SessionToken {
        /// Security token.
        token: String,
        /// Unencrypted PEM private key.
        private_key_pem: String,
    },
    /// A session token profile whose files are re-read on every request so
    /// `oci session refresh` is picked up without a restart.
    SessionTokenFile {
        /// Path to the security token file.
        token_file: PathBuf,
        /// Path to the PEM private key.
        key_file: PathBuf,
    },
    /// Resource principal v2.2 from `OCI_RESOURCE_PRINCIPAL_*` environment
    /// variables (Functions, Container Instances, and similar).
    ResourcePrincipal,
    /// Instance principal via IMDS and the X.509 federation endpoint. Needs
    /// the `oci-principals` cargo feature, else [`Error::Unsupported`].
    InstancePrincipal,
    /// OKE workload identity via the cluster proxymux. Needs the
    /// `oci-principals` cargo feature, else [`Error::Unsupported`].
    OkeWorkloadIdentity,
    /// A host-supplied token source.
    TokenSource(Arc<dyn OciTokenSource>),
}

impl std::fmt::Debug for OciAuth {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey { .. } => formatter.write_str("OciAuth::ApiKey([redacted])"),
            Self::UserPrincipal {
                tenancy,
                user,
                fingerprint,
                ..
            } => formatter
                .debug_struct("OciAuth::UserPrincipal")
                .field("tenancy", tenancy)
                .field("user", user)
                .field("fingerprint", fingerprint)
                .field("private_key_pem", &"[redacted]")
                .finish(),
            Self::SessionToken { .. } => formatter.write_str("OciAuth::SessionToken([redacted])"),
            Self::SessionTokenFile {
                token_file,
                key_file,
            } => formatter
                .debug_struct("OciAuth::SessionTokenFile")
                .field("token_file", token_file)
                .field("key_file", key_file)
                .finish(),
            Self::ResourcePrincipal => formatter.write_str("OciAuth::ResourcePrincipal"),
            Self::InstancePrincipal => formatter.write_str("OciAuth::InstancePrincipal"),
            Self::OkeWorkloadIdentity => formatter.write_str("OciAuth::OkeWorkloadIdentity"),
            Self::TokenSource(source) => formatter
                .debug_tuple("OciAuth::TokenSource")
                .field(source)
                .finish(),
        }
    }
}

/// Resolved auth: a bearer header or a signing key source.
#[derive(Debug)]
pub(crate) enum ResolvedAuth {
    Bearer(String),
    Signed(KeySource),
}

impl OciAuth {
    /// Resolves the configuration into something that can authenticate.
    ///
    /// Static keys are parsed (and validated) here so configuration mistakes
    /// surface at build time rather than on the first request.
    pub(crate) fn resolve(&self) -> Result<ResolvedAuth> {
        match self {
            Self::ApiKey { bearer } => {
                if bearer.trim().is_empty() {
                    return Err(Error::Validation("OCI API key is empty".into()));
                }
                Ok(ResolvedAuth::Bearer(bearer.clone()))
            }
            Self::UserPrincipal {
                tenancy,
                user,
                fingerprint,
                private_key_pem,
            } => Ok(ResolvedAuth::Signed(KeySource::Static {
                key_id: format!("{tenancy}/{user}/{fingerprint}"),
                key: Arc::new(SigningKey::from_pem(private_key_pem)?),
            })),
            Self::SessionToken {
                token,
                private_key_pem,
            } => Ok(ResolvedAuth::Signed(KeySource::Static {
                key_id: format!("ST${token}"),
                key: Arc::new(SigningKey::from_pem(private_key_pem)?),
            })),
            Self::SessionTokenFile {
                token_file,
                key_file,
            } => Ok(dynamic(Arc::new(FileSessionSource {
                token_file: token_file.clone(),
                key_file: key_file.clone(),
            }))),
            Self::ResourcePrincipal => {
                ResourcePrincipalSource::validate_env()?;
                Ok(dynamic(Arc::new(ResourcePrincipalSource)))
            }
            Self::InstancePrincipal => Ok(dynamic(super::principals::instance_source()?)),
            Self::OkeWorkloadIdentity => Ok(dynamic(super::principals::oke_source()?)),
            Self::TokenSource(source) => Ok(dynamic(Arc::clone(source))),
        }
    }

    /// Region advertised by the credential itself, when it carries one
    /// (resource principal).
    pub fn implied_region(&self) -> Option<String> {
        match self {
            Self::ResourcePrincipal => std::env::var(RP_REGION_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty()),
            Self::InstancePrincipal => super::principals::instance_region(),
            Self::OkeWorkloadIdentity => super::principals::oke_region(),
            _ => None,
        }
    }
}

fn dynamic(source: Arc<dyn OciTokenSource>) -> ResolvedAuth {
    ResolvedAuth::Signed(KeySource::Dynamic {
        source,
        cache: Mutex::new(None),
    })
}

/// Where the signing key and `keyId` come from for each request.
pub(crate) enum KeySource {
    /// Fixed key and key id.
    Static {
        key_id: String,
        key: Arc<SigningKey>,
    },
    /// Token source consulted on every request; the parsed key is cached
    /// until the PEM changes.
    Dynamic {
        source: Arc<dyn OciTokenSource>,
        cache: Mutex<Option<(String, Arc<SigningKey>)>>,
    },
}

impl std::fmt::Debug for KeySource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static { .. } => formatter.write_str("KeySource::Static([redacted])"),
            Self::Dynamic { source, .. } => formatter
                .debug_tuple("KeySource::Dynamic")
                .field(source)
                .finish(),
        }
    }
}

impl KeySource {
    /// The `keyId` and key to sign the next request with.
    pub(crate) fn current(&self) -> Result<(String, Arc<SigningKey>)> {
        match self {
            Self::Static { key_id, key } => Ok((key_id.clone(), Arc::clone(key))),
            Self::Dynamic { source, cache } => {
                let material = source.fetch()?;
                let mut guard = cache
                    .lock()
                    .map_err(|_| Error::Model("OCI key cache lock poisoned".into()))?;
                let key = match guard.as_ref() {
                    Some((pem, key)) if *pem == material.private_key_pem => Arc::clone(key),
                    _ => {
                        let key = Arc::new(SigningKey::from_pem(&material.private_key_pem)?);
                        *guard = Some((material.private_key_pem.clone(), Arc::clone(&key)));
                        key
                    }
                };
                Ok((format!("ST${}", material.token.trim()), key))
            }
        }
    }
}

fn read_trimmed(path: &std::path::Path, what: &str) -> Result<String> {
    std::fs::read_to_string(path)
        .map(|text| text.trim().to_string())
        .map_err(|error| {
            Error::Validation(format!(
                "could not read OCI {what} file {}: {}",
                path.display(),
                error.kind()
            ))
        })
}

/// Session token and key files re-read on every use.
#[derive(Debug)]
struct FileSessionSource {
    token_file: PathBuf,
    key_file: PathBuf,
}

impl OciTokenSource for FileSessionSource {
    fn fetch(&self) -> Result<OciSessionMaterial> {
        Ok(OciSessionMaterial {
            token: read_trimmed(&self.token_file, "session token")?,
            private_key_pem: std::fs::read_to_string(&self.key_file).map_err(|error| {
                Error::Validation(format!(
                    "could not read OCI key file {}: {}",
                    self.key_file.display(),
                    error.kind()
                ))
            })?,
        })
    }
}

/// Resource principal v2.2 read from the environment, following the OCI Go
/// SDK `common/auth` package: each value is a file path when it is an
/// absolute path, otherwise the literal material. Files are re-read per use
/// because the platform rotates them.
#[derive(Debug)]
struct ResourcePrincipalSource;

impl ResourcePrincipalSource {
    fn validate_env() -> Result<()> {
        match std::env::var(RP_VERSION_ENV).as_deref() {
            Ok("2.2") => {}
            Ok(other) => {
                return Err(Error::Unsupported(format!(
                    "OCI resource principal version `{other}` is not supported (only 2.2)"
                )));
            }
            Err(_) => {
                return Err(Error::Validation(format!("{RP_VERSION_ENV} is not set")));
            }
        }
        if std::env::var(RP_PASSPHRASE_ENV).is_ok_and(|value| !value.is_empty()) {
            return Err(Error::Unsupported(
                "encrypted resource principal private keys are not supported".into(),
            ));
        }
        for name in [RP_RPST_ENV, RP_PRIVATE_PEM_ENV, RP_REGION_ENV] {
            if std::env::var(name).is_err() {
                return Err(Error::Validation(format!("{name} is not set")));
            }
        }
        Ok(())
    }

    fn env_material(name: &str) -> Result<String> {
        let value =
            std::env::var(name).map_err(|_| Error::Validation(format!("{name} is not set")))?;
        if std::path::Path::new(&value).is_absolute() {
            std::fs::read_to_string(&value).map_err(|error| {
                Error::Validation(format!("could not read {name} file: {}", error.kind()))
            })
        } else {
            Ok(value)
        }
    }
}

impl OciTokenSource for ResourcePrincipalSource {
    fn fetch(&self) -> Result<OciSessionMaterial> {
        Ok(OciSessionMaterial {
            token: Self::env_material(RP_RPST_ENV)?.trim().to_string(),
            private_key_pem: Self::env_material(RP_PRIVATE_PEM_ENV)?,
        })
    }
}

#[cfg(test)]
#[path = "auth_tests.rs"]
mod tests;
