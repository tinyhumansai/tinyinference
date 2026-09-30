//! [`EnvSource`]: environment variables as an injected dependency.

use std::fmt::Debug;

/// Reads environment variables through a port, so no test (and no library
/// code) touches the process environment.
///
/// Implementations must not be used to log variable *values*; the hub only asks
/// for a name it was configured with.
pub trait EnvSource: Send + Sync + Debug {
    /// The value of `name`, if set.
    fn var(&self, name: &str) -> Option<String>;
}
