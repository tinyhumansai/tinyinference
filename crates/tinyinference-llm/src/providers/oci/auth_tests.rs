use super::*;
use crate::providers::oci::fixtures::test_pkcs8_pem;

#[derive(Debug)]
struct FixedSource(OciSessionMaterial);

impl OciTokenSource for FixedSource {
    fn fetch(&self) -> Result<OciSessionMaterial> {
        Ok(self.0.clone())
    }
}

fn key_id(auth: &OciAuth) -> String {
    match auth.resolve().unwrap() {
        ResolvedAuth::Signed(source) => source.current().unwrap().0,
        ResolvedAuth::Bearer(_) => panic!("expected signed auth"),
    }
}

#[test]
fn api_key_resolves_to_bearer() {
    let auth = OciAuth::ApiKey {
        bearer: "sk-test".into(),
    };
    assert!(matches!(auth.resolve().unwrap(), ResolvedAuth::Bearer(key) if key == "sk-test"));
    assert!(OciAuth::ApiKey { bearer: " ".into() }.resolve().is_err());
}

#[test]
fn user_principal_key_id_is_tenancy_user_fingerprint() {
    let auth = OciAuth::UserPrincipal {
        tenancy: "ocid1.tenancy.x".into(),
        user: "ocid1.user.y".into(),
        fingerprint: "aa:bb".into(),
        private_key_pem: test_pkcs8_pem().into(),
    };
    assert_eq!(key_id(&auth), "ocid1.tenancy.x/ocid1.user.y/aa:bb");
}

#[test]
fn session_token_key_id_has_st_prefix() {
    let auth = OciAuth::SessionToken {
        token: "abc".into(),
        private_key_pem: test_pkcs8_pem().into(),
    };
    assert_eq!(key_id(&auth), "ST$abc");
}

#[test]
fn token_source_is_consulted_and_trimmed() {
    let auth = OciAuth::TokenSource(Arc::new(FixedSource(OciSessionMaterial {
        token: " rpst \n".into(),
        private_key_pem: test_pkcs8_pem().into(),
    })));
    assert_eq!(key_id(&auth), "ST$rpst");
}

#[test]
fn session_token_file_is_reread_on_each_use() {
    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("token");
    let key = dir.path().join("key.pem");
    std::fs::write(&key, test_pkcs8_pem()).unwrap();
    std::fs::write(&token, "first\n").unwrap();
    let auth = OciAuth::SessionTokenFile {
        token_file: token.clone(),
        key_file: key,
    };
    let ResolvedAuth::Signed(source) = auth.resolve().unwrap() else {
        panic!("expected signed auth");
    };
    assert_eq!(source.current().unwrap().0, "ST$first");
    std::fs::write(&token, "second").unwrap();
    assert_eq!(source.current().unwrap().0, "ST$second");
}

#[cfg(not(feature = "oci-principals"))]
#[test]
fn unimplemented_flows_report_unsupported() {
    for auth in [OciAuth::InstancePrincipal, OciAuth::OkeWorkloadIdentity] {
        assert!(matches!(auth.resolve(), Err(Error::Unsupported(_))));
    }
}

#[test]
fn bad_static_key_fails_at_resolve_time() {
    let auth = OciAuth::SessionToken {
        token: "t".into(),
        private_key_pem: "garbage".into(),
    };
    assert!(auth.resolve().is_err());
}

#[test]
fn debug_redacts_secrets() {
    let rendered = format!(
        "{:?} {:?}",
        OciAuth::ApiKey {
            bearer: "sk-secret".into()
        },
        OciAuth::SessionToken {
            token: "tok-secret".into(),
            private_key_pem: "pem-secret".into()
        }
    );
    assert!(!rendered.contains("secret"), "{rendered}");
}
