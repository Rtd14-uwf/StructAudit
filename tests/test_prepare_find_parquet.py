import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "prepare_find.py"


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_parquet_input_real_columns(tmp_path):
    """End-to-end: a local .parquet file shaped like the real
    kensho/FIND schema (problem_text, evidence_dicts as list<struct>,
    error_type) converts correctly, including unpacking the numpy.ndarray
    that pandas produces for the nested evidence_dicts column.
    """
    raw = tmp_path / "validation-00000-of-00001.parquet"
    _write_parquet(raw, [
        {
            "doc_id": "doc1",
            "source": "BLS",
            "problem_text": "Employment rose. See Table 3 for hours worked.",
            "evidence_dicts": [{"text": "See Table 3 for hours worked."}],
            "error_type": "reference_mismatch",
            "description": "Table 3 lacks hours-worked data.",
        },
    ])
    out = tmp_path / "processed.jsonl"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(raw), "--output", str(out), "--split", "validation"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(out.read_text().strip())
    assert record["doc_id"] == "doc1"
    assert record["document_text"] == "Employment rose. See Table 3 for hours worked."
    assert record["evidence"] == ["See Table 3 for hours worked."]
    assert record["error_type"] == "reference_mismatch"
    assert record["split"] == "validation"


def test_parquet_inspect_mode_reports_ndarray_type(tmp_path):
    raw = tmp_path / "test-00000-of-00001.parquet"
    _write_parquet(raw, [{
        "doc_id": "doc1", "source": "BLS", "problem_text": "x",
        "evidence_dicts": [{"text": "x"}], "error_type": "y", "description": "z",
    }])
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(raw), "--inspect"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ndarray" in result.stdout


def test_parquet_unknown_schema_fails(tmp_path):
    raw = tmp_path / "weird.parquet"
    _write_parquet(raw, [{"totally_unexpected_column": "x"}])
    out = tmp_path / "processed.jsonl"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(raw), "--output", str(out), "--split", "test"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Available columns" in result.stderr
    assert not out.exists()


def test_jsonl_input_unchanged(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({
        "doc_id": "doc1", "source": "BLS", "document": "text",
        "evidence": ["text"], "description": "desc",
    }) + "\n")
    out = tmp_path / "processed.jsonl"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(raw), "--output", str(out), "--split", "test"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(out.read_text().strip())
    assert record["doc_id"] == "doc1"
