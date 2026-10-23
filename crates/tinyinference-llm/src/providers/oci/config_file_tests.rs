use super::*;
use crate::providers::oci::fixtures::test_pkcs8_pem;

fn setup() -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join(".oci")).unwrap();
    std::fs::write(dir.path().join(".oci/key.pem"), test_pkcs8_pem()).unwrap();
    let config = "\
# comment
[DEFAULT]
user=ocid1.user.default
fingerprint=aa:bb:cc
tenancy=ocid1.tenancy.default
region=us-ashburn-1
key_file=~/.oci/key.pem

[OTHER]
region = us-chicago-1
user = ocid1.user.other

[SESSION]
fingerprint=dd
key_file=~/.oci/key.pem
security_token_file=~/.oci/token
region=eu-frankfurt-1

[ENCRYPTED]
pass_phrase=hunter2
"
    .to_string();
    (dir, config)
}

#[test]
fn default_profile_resolves_to_user_principal() {
    let (dir, config) = setup();
    let profile = parse_profile(&config, "DEFAULT", Some(dir.path())).unwrap();
    assert_eq!(profile.region.as_deref(), Some("us-ashburn-1"));
    assert_eq!(profile.tenancy.as_deref(), Some("ocid1.tenancy.default"));
    match profile.auth {
        OciAuth::UserPrincipal {
            tenancy,
            user,
            fingerprint,
            private_key_pem,
        } => {
            assert_eq!(tenancy, "ocid1.tenancy.default");
            assert_eq!(user, "ocid1.user.default");
            assert_eq!(fingerprint, "aa:bb:cc");
            assert_eq!(private_key_pem, test_pkcs8_pem());
        }
        other => panic!("unexpected auth {other:?}"),
    }
}

#[test]
fn named_profile_inherits_from_default_and_overrides() {
    let (dir, config) = setup();
    let profile = parse_profile(&config, "OTHER", Some(dir.path())).unwrap();
    assert_eq!(profile.region.as_deref(), Some("us-chicago-1"));
    let OciAuth::UserPrincipal {
        user,
        fingerprint,
        tenancy,
        ..
    } = profile.auth
    else {
        panic!("expected user principal");
    };
    assert_eq!(user, "ocid1.user.other");
    assert_eq!(fingerprint, "aa:bb:cc");
    assert_eq!(tenancy, "ocid1.tenancy.default");
}

#[test]
fn security_token_profile_becomes_a_session_token_file_auth() {
    let (dir, config) = setup();
    let profile = parse_profile(&config, "SESSION", Some(dir.path())).unwrap();
    assert_eq!(profile.region.as_deref(), Some("eu-frankfurt-1"));
    match profile.auth {
        OciAuth::SessionTokenFile {
            token_file,
            key_file,
        } => {
            assert_eq!(token_file, dir.path().join(".oci/token"));
            assert_eq!(key_file, dir.path().join(".oci/key.pem"));
        }
        other => panic!("unexpected auth {other:?}"),
    }
}

#[test]
fn missing_profile_and_passphrase_are_errors() {
    let (dir, config) = setup();
    assert!(parse_profile(&config, "NOPE", Some(dir.path())).is_err());
    let error = parse_profile(&config, "ENCRYPTED", Some(dir.path())).unwrap_err();
    assert!(error.to_string().contains("pass_phrase"));
}

#[test]
fn missing_required_key_is_named_in_the_error() {
    let config = "[DEFAULT]\nkey_file=/nonexistent\n";
    let error = parse_profile(config, "DEFAULT", None)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("key file") || error.contains("missing"),
        "{error}"
    );
}

#[test]
fn tilde_expansion_only_applies_with_a_home() {
    assert_eq!(
        expand_home("~/a", Some(Path::new("/h"))),
        PathBuf::from("/h/a")
    );
    assert_eq!(expand_home("~/a", None), PathBuf::from("~/a"));
    assert_eq!(
        expand_home("/abs", Some(Path::new("/h"))),
        PathBuf::from("/abs")
    );
}
