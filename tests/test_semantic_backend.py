from structaudit.models.semantic_backend import LexicalOverlapBackend


def test_reference_topic_support_clear_match():
    backend = LexicalOverlapBackend()
    result = backend.classify("reference_topic_support", {
        "expected_topic": "refinancing defaults",
        "resolved_target_text": "Table 7: Refinancing defaults by quarter",
    })
    assert result["label"] == "SUPPORTED"


def test_reference_topic_support_clear_mismatch():
    backend = LexicalOverlapBackend()
    result = backend.classify("reference_topic_support", {
        "expected_topic": "refinancing defaults",
        "resolved_target_text": "Quarterly headcount summary by region",
    })
    assert result["label"] == "NOT_SUPPORTED"


def test_reference_topic_support_unsupported_task_raises():
    backend = LexicalOverlapBackend()
    try:
        backend.classify("not_a_real_task", {})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_call_log_records_required_fields():
    backend = LexicalOverlapBackend()
    backend.classify("reference_topic_support", {
        "expected_topic": "revenue growth", "resolved_target_text": "Revenue growth by segment",
    })
    assert len(backend.call_log) == 1
    entry = backend.call_log[0]
    assert entry.task == "reference_topic_support"
    assert entry.model_backend == "local"
    assert entry.model_name == "lexical-overlap-v1"
    assert entry.parsed_output["label"] == "SUPPORTED"
    assert entry.latency >= 0.0


def test_call_log_accumulates_across_calls():
    backend = LexicalOverlapBackend()
    backend.classify("reference_topic_support", {"expected_topic": "a", "resolved_target_text": "a"})
    backend.classify("reference_topic_support", {"expected_topic": "b", "resolved_target_text": "b"})
    assert len(backend.call_log) == 2
