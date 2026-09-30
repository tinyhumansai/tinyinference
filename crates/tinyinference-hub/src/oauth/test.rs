//! Tests that every OAuth entry point is a typed refusal.

use crate::error::{HubError, Operation};
use crate::hub::fixtures::Bed;
use crate::ids::KindId;

#[tokio::test]
async fn sim_oauth_disabled_every_flow_is_unsupported() {
    let bed = Bed::new();
    for kind in [
        "openai-codex",
        "openrouter",
        "gemini-code-assist",
        "anything",
    ] {
        let kind = KindId::new(kind);
        assert!(matches!(
            bed.hub.oauth_start(&bed.scope, &kind).await,
            Err(HubError::Unsupported { op: Operation::OAuth, kind: k }) if k == kind
        ));
        assert!(matches!(
            bed.hub.oauth_complete(&bed.scope, &kind, "code").await,
            Err(HubError::Unsupported {
                op: Operation::OAuth,
                ..
            })
        ));
    }
    assert_eq!(bed.ports.http.request_count(), 0);
}

#[test]
fn oauth_feature_types_only_and_debug_never_prints_a_credential() {
    let grant = super::OAuthGrant {
        credential: crate::secret::Secret::new("sk-not-a-real-key"),
    };
    assert!(!format!("{grant:?}").contains("sk-not-a-real-key"));
    let start = super::OAuthStart {
        authorize_url: "https://login.test/authorize".into(),
        state: "s".into(),
    };
    assert_eq!(start.clone(), start);
}
