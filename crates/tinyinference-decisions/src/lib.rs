//! Typed Rust access to hosted and local Jev-compatible decision models and
//! Levanto Sage.
//!
//! A request supplies text or structured state plus independent [`Question`]s.
//! Jev returns typed choices, ordinal scores, and yes/no probabilities for code
//! to compose. This crate validates both sides of that wire contract and owns
//! only the HTTP wait; policy, thresholds, and actions stay with the caller.
//!
//! # Example
//!
//! ```no_run
//! use std::collections::BTreeMap;
//! use serde_json::json;
//! use tinyinference_decisions::{Choice, Client, EvaluationRequest, Question};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let criteria = BTreeMap::from([
//!     ("billing".to_owned(), Some(json!("payments and refunds"))),
//!     ("technical".to_owned(), Some(json!("bugs and outages"))),
//! ]);
//! let request = EvaluationRequest::jev(
//!     json!({"ticket": "I was charged twice"}),
//!     BTreeMap::from([(
//!         "route".to_owned(),
//!         Question::Choice(Choice {
//!             instructions: json!("Which team should handle this ticket?"),
//!             criteria,
//!         }),
//!     )]),
//! );
//! let result = Client::from_env()?.evaluate(&request).await?;
//! let answer = result.response.answers.get("route").ok_or_else(|| {
//!     std::io::Error::new(std::io::ErrorKind::InvalidData, "response omitted route answer")
//! })?;
//! println!("{answer:?}");
//! # Ok(())
//! # }
//! ```
//!
//! [`sage`] contains Levanto Sage's Yes/No, Choice, Scale, Sort, and Tags API.
//! Use [`ClientConfig::self_hosted`] with [`EvaluationRequest::new`] to target
//! a user-managed OpenJEV or other Jev-compatible decisions endpoint with an
//! explicit model id and optional bearer authentication.
//! This crate does not execute selected actions or infer permission from a
//! decision probability.

mod client;
mod error;
mod request;
mod response;
pub mod sage;

pub use client::{
    Client, ClientConfig, EvaluationFailure, EvaluationResult, Provider, RetryPolicy,
};
pub use error::{Error, Result};
pub use request::{
    Asset, BoxQuestion, Choice, EvaluationRequest, ImageAsset, ImageFormat, Noul, NoulCriteria,
    Question, Score,
};
pub use response::{
    Answer, BoxAnswer, BoxEdges, ChoiceAnswer, EvaluationResponse, NoulAnswer, ScoreAnswer, Usage,
};
