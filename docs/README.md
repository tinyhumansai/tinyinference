# TinyInference Documentation

- [`specs/README.md`](specs/README.md) records accepted behavior and ownership
  boundaries.
- [`plans/README.md`](plans/README.md) indexes implementation plans when a
  change is large enough to need one.
- [`adr/`](adr/) holds immutable architecture decisions.

Most API documentation lives next to the implementation as compiled rustdoc.
- [`tinyinference-decisions.md`](tinyinference-decisions.md) documents the Jev and Sage decision API crate.

- [Shared provider-call budgets](budgets.md): atomic reservations, child ledgers,
  conservative unknown spend and transport retry admission.
