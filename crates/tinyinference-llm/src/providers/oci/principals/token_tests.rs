use super::*;
use crate::providers::oci::principals::mock_http::jwt;
use std::cell::Cell;

fn material(token: String) -> OciSessionMaterial {
    OciSessionMaterial {
        token,
        private_key_pem: "pem".into(),
    }
}

#[test]
fn parses_exp_and_iat_from_a_jwt() {
    let times = parse_token_times(&jwt(100, 4000)).unwrap();
    assert_eq!(
        times,
        TokenTimes {
            iat: Some(100),
            exp: 4000
        }
    );
}

#[test]
fn rejects_malformed_tokens() {
    for token in ["", "a.b", "a.b.c.d", "a.!!!.c", "a.e30.c"] {
        assert!(parse_token_times(token).is_err(), "{token}");
    }
}

#[test]
fn validity_honours_the_five_minute_buffer() {
    let times = TokenTimes {
        iat: None,
        exp: 10_000,
    };
    assert!(times.valid_at(10_000 - REFRESH_BUFFER_SECS - 1, false));
    assert!(!times.valid_at(10_000 - REFRESH_BUFFER_SECS, false));
    assert!(!times.valid_at(20_000, false));
}

#[test]
fn soft_validity_retires_tokens_at_half_life() {
    let times = TokenTimes {
        iat: Some(0),
        exp: 7_200,
    };
    assert!(times.valid_at(3_000, true));
    assert!(!times.valid_at(3_400, true));
    assert!(times.valid_at(3_400, false));
}

#[test]
fn cache_reuses_until_expiry_then_renews() {
    let cache = TokenCache::default();
    let calls = Cell::new(0);
    let renew = |exp: i64| {
        calls.set(calls.get() + 1);
        Ok(material(jwt(0, exp)))
    };
    cache.get_or_renew(1_000, false, || renew(10_000)).unwrap();
    cache.get_or_renew(2_000, false, || renew(10_000)).unwrap();
    assert_eq!(calls.get(), 1);
    cache.get_or_renew(9_800, false, || renew(20_000)).unwrap();
    assert_eq!(calls.get(), 2);
    cache.get_or_renew(10_000, false, || renew(20_000)).unwrap();
    assert_eq!(calls.get(), 2);
}

#[test]
fn failed_renewal_is_not_cached() {
    let cache = TokenCache::default();
    assert!(
        cache
            .get_or_renew(0, false, || Err(Error::Model("boom".into())))
            .is_err()
    );
    assert!(
        cache
            .get_or_renew(0, false, || Ok(material(jwt(0, 100_000))))
            .is_ok()
    );
}

#[tokio::test]
async fn block_on_works_inside_a_runtime() {
    let value = block_on(async { Ok::<_, Error>(7) }).unwrap();
    assert_eq!(value, 7);
}
