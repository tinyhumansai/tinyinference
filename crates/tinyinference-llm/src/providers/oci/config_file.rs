//! Parser for the OCI CLI/SDK config file (`~/.oci/config`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::auth::OciAuth;
use crate::{Error, Result};

/// What a config-file profile resolves to.
#[derive(Debug, Clone)]
pub struct OciProfile {
    /// Credentials for the profile.
    pub auth: OciAuth,
    /// The profile's `region`, when set.
    pub region: Option<String>,
    /// The profile's `tenancy`, when set (useful as a compartment default).
    pub tenancy: Option<String>,
}

/// Default config location: `$OCI_CLI_CONFIG_FILE`, else `~/.oci/config`.
pub fn default_config_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("OCI_CLI_CONFIG_FILE")
        && !path.trim().is_empty()
    {
        return Some(PathBuf::from(path));
    }
    std::env::var_os("HOME").map(|home| Path::new(&home).join(".oci").join("config"))
}

/// Reads `path` and resolves `profile` (usually `DEFAULT`).
///
/// # Errors
///
/// [`Error::Validation`] when the file or profile is missing or incomplete.
pub fn load_profile(path: &Path, profile: &str) -> Result<OciProfile> {
    let contents = std::fs::read_to_string(path).map_err(|error| {
        Error::Validation(format!(
            "could not read OCI config {}: {}",
            path.display(),
            error.kind()
        ))
    })?;
    let home = std::env::var_os("HOME").map(PathBuf::from);
    parse_profile(&contents, profile, home.as_deref())
}

/// Parses config `contents` and resolves `profile`; `home` expands a leading
/// `~` in file paths. Keys from `[DEFAULT]` are inherited unless the profile
/// overrides them. The private key file is read here for API-key profiles; a
/// session-token profile keeps its file paths so refreshes are honoured.
///
/// # Errors
///
/// [`Error::Validation`] for a missing profile, missing required keys, or a
/// passphrase-protected key.
pub fn parse_profile(contents: &str, profile: &str, home: Option<&Path>) -> Result<OciProfile> {
    let sections = parse_sections(contents);
    let selected = sections.get(profile);
    if selected.is_none() && profile != "DEFAULT" {
        return Err(Error::Validation(format!(
            "OCI config has no profile named `{profile}`"
        )));
    }
    let mut values: HashMap<String, String> = sections.get("DEFAULT").cloned().unwrap_or_default();
    if let Some(selected) = selected {
        values.extend(selected.clone());
    }
    let get = |key: &str| values.get(key).filter(|value| !value.is_empty()).cloned();
    let need = |key: &str| {
        get(key)
            .ok_or_else(|| Error::Validation(format!("OCI profile `{profile}` is missing `{key}`")))
    };
    if get("pass_phrase").is_some() {
        return Err(Error::Validation(
            "OCI profiles with a key pass_phrase are not supported; use an unencrypted key".into(),
        ));
    }
    let key_file = expand_home(&need("key_file")?, home);
    let auth = if let Some(token_file) = get("security_token_file") {
        OciAuth::SessionTokenFile {
            token_file: expand_home(&token_file, home),
            key_file,
        }
    } else {
        let private_key_pem = std::fs::read_to_string(&key_file).map_err(|error| {
            Error::Validation(format!(
                "could not read OCI key file {}: {}",
                key_file.display(),
                error.kind()
            ))
        })?;
        OciAuth::UserPrincipal {
            tenancy: need("tenancy")?,
            user: need("user")?,
            fingerprint: need("fingerprint")?,
            private_key_pem,
        }
    };
    Ok(OciProfile {
        auth,
        region: get("region"),
        tenancy: get("tenancy"),
    })
}

fn expand_home(path: &str, home: Option<&Path>) -> PathBuf {
    match (path.strip_prefix("~/"), home) {
        (Some(rest), Some(home)) => home.join(rest),
        _ if path == "~" => home.map_or_else(|| PathBuf::from(path), Path::to_path_buf),
        _ => PathBuf::from(path),
    }
}

fn parse_sections(contents: &str) -> HashMap<String, HashMap<String, String>> {
    let mut sections: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut current: Option<String> = None;
    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(name) = line
            .strip_prefix('[')
            .and_then(|rest| rest.strip_suffix(']'))
        {
            let name = name.trim().to_string();
            sections.entry(name.clone()).or_default();
            current = Some(name);
        } else if let (Some(section), Some((key, value))) = (&current, line.split_once('=')) {
            sections
                .entry(section.clone())
                .or_default()
                .insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    sections
}

#[cfg(test)]
#[path = "config_file_tests.rs"]
mod tests;
