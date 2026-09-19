//! Process-wide guard that lets a host forbid network-backed model calls.
//!
//! Tests and evaluation harnesses often want a hard guarantee that a run
//! never reaches the network, regardless of which model a caller happened to
//! configure. [`deny_network_models`] sets a process-wide flag that
//! network-backed [`crate::model::ChatModel`] adapters (the OpenAI and
//! Anthropic providers) check before issuing any HTTP request, returning
//! [`crate::Error::Validation`] instead of dialing out. [`MockModel`](crate::providers::MockModel)
//! and other in-process adapters are unaffected because they never reach this
//! check.
//!
//! The guard is a single process-wide `AtomicBool`. It is intended for test
//! setup (for example a `#[ctor]`-style fixture or the first line of a test
//! module) rather than per-request policy.

use std::sync::atomic::{AtomicBool, Ordering};

static NETWORK_MODELS_DENIED: AtomicBool = AtomicBool::new(false);

/// Forbids network-backed model providers from issuing HTTP requests for the
/// remainder of the process.
///
/// Idempotent: calling this more than once has no additional effect. Use
/// [`allow_network_models`] to lift the restriction (primarily useful for
/// resetting shared test state between cases).
pub fn deny_network_models() {
    NETWORK_MODELS_DENIED.store(true, Ordering::SeqCst);
}

/// Lifts a restriction previously installed by [`deny_network_models`].
pub fn allow_network_models() {
    NETWORK_MODELS_DENIED.store(false, Ordering::SeqCst);
}

/// Returns whether [`deny_network_models`] is currently in effect.
#[must_use]
pub fn network_models_denied() -> bool {
    NETWORK_MODELS_DENIED.load(Ordering::SeqCst)
}

/// Returns [`crate::Error::Validation`] when network models are denied,
/// otherwise `Ok(())`. Network-backed provider adapters call this at the top
/// of every request-issuing path.
pub fn ensure_network_models_allowed() -> crate::Result<()> {
    if network_models_denied() {
        return Err(crate::Error::Validation(
            "network-backed model calls are denied for this process; call \
             tinyinference_llm::allow_network_models() to lift the restriction"
                .to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use std::sync::Mutex;

    // The guard is process-wide `static` state, so tests that flip it must
    // not interleave with each other; serialize them with a mutex rather
    // than relying on cargo test's default single-process, multi-thread
    // execution to happen to avoid collisions.
    static GUARD_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn denies_and_allows_round_trip() {
        let _lock = GUARD_TEST_LOCK.lock().unwrap();
        allow_network_models();
        assert!(!network_models_denied());
        assert!(ensure_network_models_allowed().is_ok());

        deny_network_models();
        assert!(network_models_denied());
        assert!(ensure_network_models_allowed().is_err());

        allow_network_models();
        assert!(!network_models_denied());
        assert!(ensure_network_models_allowed().is_ok());
    }

    #[test]
    fn deny_is_idempotent() {
        let _lock = GUARD_TEST_LOCK.lock().unwrap();
        deny_network_models();
        deny_network_models();
        assert!(network_models_denied());
        allow_network_models();
    }
}
