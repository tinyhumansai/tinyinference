use super::*;

use crate::model::ModelResponse;
use serde_json::json;

#[test]
fn old_responses_read_without_inventing_execution_or_rich_output() {
    let response: ModelResponse = serde_json::from_value(json!({
        "message":{"content":[{"text":"hello"}],"tool_calls":[]}
    }))
    .unwrap();
    assert!(response.output.is_empty());
    assert!(response.execution.is_none());
    assert_eq!(response.text(), "hello");
    let encoded = serde_json::to_value(response).unwrap();
    assert!(encoded.get("execution").is_none());
    assert!(encoded.get("output").is_none());
}

#[test]
fn future_statuses_survive_serialization_without_becoming_success() {
    let status: ExecutionStatus = serde_json::from_value(json!("waiting_for_review")).unwrap();
    assert!(!status.is_terminal());
    assert_eq!(
        serde_json::to_value(status).unwrap(),
        json!("waiting_for_review")
    );
}

#[test]
fn deferred_cancellation_serializes_with_or_without_a_snapshot() {
    use crate::model::DeferredStatus;
    for response in [None, Some(Box::new(ModelResponse::assistant("partial")))] {
        let status = DeferredStatus::Cancelled { response };
        let value = serde_json::to_value(status).unwrap();
        assert_eq!(value["status"], "cancelled");
        assert!(matches!(
            serde_json::from_value::<DeferredStatus>(value).unwrap(),
            DeferredStatus::Cancelled { .. }
        ));
    }
}

#[test]
fn reconstruction_preserves_progress_supplied_in_execution_snapshots() {
    use crate::model::{ModelStreamItem, StreamAccumulator};
    let reported = ModelProgress::ReasoningPhase { active: true };
    let later = ModelProgress::ReasoningText {
        text: "working".into(),
    };
    for snapshot_progress in [vec![reported.clone()], Vec::new()] {
        for separate_progress in [vec![later.clone()], Vec::new()] {
            let execution = ModelExecution {
                id: "resp_1".into(),
                model: "provider/model".into(),
                status: ExecutionStatus::InProgress,
                incomplete_reason: None,
                sequence_number: Some(3),
                cost: None,
                tool_usage: Default::default(),
                progress: snapshot_progress.clone(),
            };
            let mut accumulator = StreamAccumulator::new();
            accumulator.push(&ModelStreamItem::OutputEvent(ModelOutputEvent::Execution(
                Box::new(execution),
            )));
            for progress in &separate_progress {
                accumulator.push(&ModelStreamItem::OutputEvent(ModelOutputEvent::Progress(
                    progress.clone(),
                )));
            }
            let response = accumulator.finish().unwrap();
            let expected = if snapshot_progress.is_empty() {
                separate_progress
            } else {
                snapshot_progress.clone()
            };
            assert_eq!(response.execution.unwrap().progress, expected);
        }
    }
}
