# Request module

This module defines the exact state-and-questions payload sent to TypeSafe's
System One endpoint. `types.rs` holds the serde wire types, `mod.rs` validates
the documented primitive bounds before transport, and `mod_tests.rs` pins both.
`image.rs` builds inline image assets and checks them against the provider's
image limits (format, side lengths, sizes) by reading only the file header.
Box questions stay typed here even for providers that cannot answer them;
the client rewrites them (see `../emulation/`) after this module validates the
caller's request.
