import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "prepare_find.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_default_field_map_succeeds(tmp_path):
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(raw, [{
        "doc_id": "doc1", "source": "BLS", "document": "text here",
        "evidence": ["text here"], "description": "desc",
    }])
    out = tmp_path / "processed.jsonl"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(raw), "--output", str(out), "--split", "validation"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(out.read_text().strip())
    assert record["doc_id"] == "doc1"
    assert record["split"] == "validation"


def test_unknown_schema_fails(tmp_path):
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(raw, [{"uid": "doc2", "body": "x", "evidence_spans": ["x"], "explanation": "y", "source": "SEC"}])
    out = tmp_path / "processed.jsonl"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(raw), "--output", str(out), "--split", "test"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Available columns" in result.stderr
    assert not out.exists()


def test_field_map_override_fixes_schema(tmp_path):
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(raw, [{"uid": "doc2", "body": "x", "evidence_spans": ["x"], "explanation": "y", "source": "SEC"}])
    field_map = tmp_path / "field_map.json"
    field_map.write_text(json.dumps({
        "doc_id": ["uid"], "document_text": ["body"],
        "evidence": ["evidence_spans"], "description": ["explanation"],
    }))
    out = tmp_path / "processed.jsonl"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(raw), "--output", str(out),
         "--split", "test", "--field-map", str(field_map)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(out.read_text().strip())
    assert record["doc_id"] == "doc2"
    assert record["source"] == "SEC"


def test_inspect_mode_writes_no_output(tmp_path):
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(raw, [{"weird_col": "x"}])
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(raw), "--inspect"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "weird_col" in result.stdout


def test_field_map_rejects_unknown_override_keys(tmp_path):
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(raw, [{"doc_id": "doc1", "source": "BLS", "document": "t", "evidence": ["t"], "description": "d"}])
    field_map = tmp_path / "field_map.json"
    field_map.write_text(json.dumps({"not_a_real_field": ["x"]}))
    out = tmp_path / "processed.jsonl"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(raw), "--output", str(out),
         "--split", "test", "--field-map", str(field_map)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "unrecognized keys" in result.stderr


def test_missing_input_file_fails_cleanly(tmp_path):
    out = tmp_path / "processed.jsonl"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(tmp_path / "does_not_exist.jsonl"),
         "--output", str(out), "--split", "test"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "not found" in result.stderr
