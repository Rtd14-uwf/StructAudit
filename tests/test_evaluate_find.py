import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from structaudit.evaluation.schema import (
    DocumentPrediction,
    GoldExample,
    PredictedInconsistencyRecord,
    hf_dataset_to_gold_examples,
    make_blind_predictions,
    read_gold_jsonl,
    read_predictions_jsonl,
    write_jsonl,
)

from scripts.evaluate_find import evaluate, per_document_evidence_score

GOLD = GoldExample(
    doc_id="doc1",
    source="BLS",
    document_text="Employment rose. See Table 3 for hours worked.",
    evidence=["See Table 3 for hours worked."],
    description="Table 3 lacks hours-worked data.",
)


def test_jsonl_roundtrip(tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    write_jsonl([GOLD], gold_path)
    loaded = read_gold_jsonl(gold_path)
    assert loaded["doc1"].evidence == GOLD.evidence
    assert loaded["doc1"].source == "BLS"


def test_read_gold_jsonl_malformed_line_raises_with_line_number(tmp_path):
    bad_path = tmp_path / "bad.jsonl"
    valid_line = GOLD.to_json()
    import json as _json
    bad_path.write_text(_json.dumps(valid_line) + "\nnot valid json\n")
    try:
        read_gold_jsonl(bad_path)
        assert False, "expected ValueError"
    except ValueError as e:
        assert ":2:" in str(e)  # the malformed line is line 2
        assert "invalid JSON" in str(e)


def test_make_blind_predictions_then_read_back(tmp_path):
    blind = make_blind_predictions([GOLD], system="toy")
    pred_path = tmp_path / "predictions.jsonl"
    write_jsonl(blind, pred_path)
    loaded = read_predictions_jsonl(pred_path)
    assert loaded["doc1"].answers == []
    assert loaded["doc1"].system == "toy"


def test_adapter_default_field_names():
    fake_hf_rows = [
        {
            "doc_id": "doc1",
            "source": "BLS",
            "document": "Employment rose. See Table 3 for hours worked.",
            "evidence": ["See Table 3 for hours worked."],
            "description": "Table 3 lacks hours-worked data.",
        }
    ]
    examples = hf_dataset_to_gold_examples(fake_hf_rows, split="validation")
    assert len(examples) == 1
    assert examples[0].doc_id == "doc1"
    assert examples[0].split == "validation"


def test_adapter_real_column_shape():
    doc = "Employment rose. See Table 3 for hours worked."
    start = doc.index("See Table 3")
    end = start + len("See Table 3 for hours worked.")
    fake_hf_rows = [
        {
            "filename": "doc1.txt",
            "dataset": "BLS",
            "problem_text": doc,
            "evidence_dicts": [{"token": doc[start:end], "start": start, "end": end}],
            "error_type": "reference_mismatch",
            "description": "Table 3 lacks hours-worked data.",
        }
    ]
    examples = hf_dataset_to_gold_examples(fake_hf_rows, split="validation")
    assert len(examples) == 1
    ex = examples[0]
    assert ex.doc_id == "doc1.txt"
    assert ex.source == "BLS"
    assert ex.document_text == doc
    assert ex.evidence == ["See Table 3 for hours worked."]
    assert ex.error_type == "reference_mismatch"


def test_evidence_dict_prefers_token_over_bad_offsets():
    doc = "Employment rose. See Table 3 for hours worked."
    fake_hf_rows = [
        {
            "filename": "doc1.txt", "dataset": "BLS", "problem_text": doc,
            # start/end deliberately wrong -- token must still be used as-is
            "evidence_dicts": [{"token": "See Table 3 for hours worked.", "start": 0, "end": 5}],
            "description": "desc",
        }
    ]
    examples = hf_dataset_to_gold_examples(fake_hf_rows, split="test")
    assert examples[0].evidence == ["See Table 3 for hours worked."]


def test_evidence_dict_offset_is_last_resort():
    doc = "short doc"
    fake_hf_rows = [
        {
            "filename": "doc1.txt", "dataset": "BLS", "problem_text": doc,
            "evidence_dicts": [{"start": 0, "end": 500}],  # no "token" key at all, and past len(doc)
            "description": "desc",
        }
    ]
    try:
        hf_dataset_to_gold_examples(fake_hf_rows, split="test")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "out of bounds" in str(e)


def test_evidence_prefers_plain_field():
    fake_hf_rows = [
        {
            "filename": "ppi.md", "dataset": "BLS", "problem_text": "doc text",
            "evidence": ["Services", "Raw materials for intermediate demand"],
            "evidence_dicts": [{"token": "Services", "start": 0, "end": 500}],  # would be out of bounds if used
            "description": "desc",
        }
    ]
    examples = hf_dataset_to_gold_examples(fake_hf_rows, split="test")
    assert examples[0].evidence == ["Services", "Raw materials for intermediate demand"]


def test_evidence_dict_unknown_key_fails():
    fake_hf_rows = [
        {
            "doc_id": "doc1", "source": "BLS", "problem_text": "x",
            "evidence_dicts": [{"unexpected_key": "See Table 3."}],
            "description": "desc",
        }
    ]
    try:
        hf_dataset_to_gold_examples(fake_hf_rows, split="test")
        assert False, "expected KeyError"
    except KeyError as e:
        assert "Available keys" in str(e)


def test_error_type_optional():
    fake_hf_rows = [
        {
            "doc_id": "doc1", "source": "BLS", "problem_text": "x",
            "evidence_dicts": ["plain string evidence works too"],
            "description": "desc",
            # no error_type column at all
        }
    ]
    examples = hf_dataset_to_gold_examples(fake_hf_rows, split="test")
    assert examples[0].error_type is None


def test_adapter_unknown_schema_fails():
    fake_hf_rows = [{"totally_unexpected_column": "x"}]
    try:
        hf_dataset_to_gold_examples(fake_hf_rows, split="test")
        assert False, "expected KeyError"
    except KeyError as e:
        assert "Available columns" in str(e)


def test_per_document_score_exact_match():
    prediction = DocumentPrediction(
        doc_id="doc1",
        answers=[PredictedInconsistencyRecord(evidence=GOLD.evidence, description="x")],
    )
    assert per_document_evidence_score(GOLD, prediction) == 1.0


def test_per_document_score_missing_prediction():
    assert per_document_evidence_score(GOLD, None) == 0.0


def test_evaluate_aggregates_by_source():
    gold_examples = {"doc1": GOLD}
    predictions = {}  # nothing predicted at all
    metrics = evaluate(gold_examples, predictions)
    assert metrics.evidence_score.by_source["BLS"] == 0.0
    assert metrics.n_missing_predictions == 1
    assert metrics.description_score.not_implemented is True


def test_cli_missing_file_fails_cleanly(tmp_path):
    import subprocess

    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_find.py"
    gold_path = tmp_path / "gold.jsonl"
    write_jsonl([GOLD], gold_path)

    result = subprocess.run(
        [sys.executable, str(script),
         "--predictions", str(tmp_path / "does_not_exist.jsonl"),
         "--gold", str(gold_path),
         "--output", str(tmp_path / "out.json")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "not found" in result.stderr


def test_cli_malformed_jsonl_fails_cleanly(tmp_path):
    import subprocess

    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_find.py"
    gold_path = tmp_path / "gold.jsonl"
    write_jsonl([GOLD], gold_path)
    bad_predictions = tmp_path / "predictions.jsonl"
    bad_predictions.write_text("not valid json\n")

    result = subprocess.run(
        [sys.executable, str(script),
         "--predictions", str(bad_predictions),
         "--gold", str(gold_path),
         "--output", str(tmp_path / "out.json")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "invalid JSON" in result.stderr
    assert ":1:" in result.stderr  # line number included
