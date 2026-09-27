"""
Data schema for the FIND evaluation harness.

Two record types flow through this harness:

  GoldExample        -- one FIND document plus its single expert-inserted
                         inconsistency (evidence spans + description).
  DocumentPrediction  -- one system's candidate inconsistencies for a
                         document, before grading against the gold answer.

Every system under evaluation should write predictions in this shared
format so the evaluator never needs to know which system produced them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass
class GoldExample:
    """One FIND document and its single expert-inserted inconsistency.
    """

    doc_id: str
    source: str  # BLS | PRE | SEC | EMM | PG | MFR | cs.CL, etc.
    document_text: str
    evidence: list[str]
    description: str
    error_type: str | None = None  # taxonomy of the inserted inconsistency
    split: str = "test"  # "test" | "validation"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> "GoldExample":
        return cls(
            doc_id=record["doc_id"],
            source=record["source"],
            document_text=record["document_text"],
            evidence=record["evidence"],
            description=record["description"],
            error_type=record.get("error_type"),
            split=record.get("split", "test"),
        )


_DEFAULT_FIELD_MAP = {
    "doc_id": ("filename", "doc_id", "id", "document_id"),
    "source": ("dataset", "source", "data_source", "dataset_source"),
    "document_text": ("problem_text", "document", "document_text", "text"),
    "evidence": ("evidence", "evidence_dicts", "evidence_spans"),
    "description": ("description", "explanation"),
    "error_type": ("error_type",),
}


_EVIDENCE_TEXT_KEYS = ("token", "text", "span", "quote", "evidence", "content")


def _extract_evidence_texts(raw_evidence: list[Any], document_text: str | None = None) -> list[str]:
    """Normalize the evidence field to list[str], handling both a plain
    list[str] and the dict-shaped "evidence_dicts" ({token, start, end}).
    """
    if not raw_evidence:
        return []

    if isinstance(raw_evidence[0], str):
        return list(raw_evidence)

    texts = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            raise TypeError(f"unexpected evidence item {item!r}, expected str or dict")

        matched_key = next((key for key in _EVIDENCE_TEXT_KEYS if key in item), None)
        if matched_key is not None:
            texts.append(str(item[matched_key]))
            continue

        has_offsets = (
            document_text is not None
            and isinstance(item.get("start"), int)
            and isinstance(item.get("end"), int)
        )
        if has_offsets:
            start, end = item["start"], item["end"]
            if not (0 <= start < end <= len(document_text)):
                raise ValueError(f"evidence_dicts start/end ({start}, {end}) out of bounds for document_text")
            texts.append(document_text[start:end])
            continue

        raise KeyError(f"no text-bearing key among {_EVIDENCE_TEXT_KEYS}. Available keys: {sorted(item.keys())}")
    return texts


def _resolve_field(record: dict[str, Any], candidates: tuple[str, ...], field_name: str) -> Any:
    for candidate in candidates:
        if candidate in record:
            return record[candidate]

    raise KeyError(f"no column for {field_name!r} among {candidates}. Available columns: {sorted(record.keys())}")


def hf_dataset_to_gold_examples(
    dataset: Iterable[dict[str, Any]],
    split: str,
    field_map: dict[str, tuple[str, ...]] | None = None,
) -> list[GoldExample]:
    """Convert HF dataset rows (or any iterable of dict-like rows) into GoldExample records"""
    field_map = field_map or _DEFAULT_FIELD_MAP
    examples = []
    for row in dataset:
        doc_id = str(_resolve_field(row, field_map["doc_id"], "doc_id"))
        source = str(_resolve_field(row, field_map["source"], "source"))
        document_text = str(_resolve_field(row, field_map["document_text"], "document_text"))
        raw_evidence = list(_resolve_field(row, field_map["evidence"], "evidence"))
        evidence = _extract_evidence_texts(raw_evidence, document_text=document_text)
        description = str(_resolve_field(row, field_map["description"], "description"))
        error_type = None
        if "error_type" in field_map:
            try:
                error_type = str(_resolve_field(row, field_map["error_type"], "error_type"))
            except KeyError:
                pass  # optional metadata

        examples.append(
            GoldExample(
                doc_id=doc_id,
                source=source,
                document_text=document_text,
                evidence=evidence,
                description=description,
                error_type=error_type,
                split=split,
            )
        )
    return examples


@dataclass
class PredictedInconsistencyRecord:
    evidence: list[str]
    description: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> "PredictedInconsistencyRecord":
        return cls(evidence=list(record["evidence"]), description=str(record["description"]))


@dataclass
class DocumentPrediction:
    """One system's candidate inconsistencies for a single document."""

    doc_id: str
    answers: list[PredictedInconsistencyRecord] = field(default_factory=list)
    system: str | None = None
    raw_output: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "answers": [a.to_json() for a in self.answers],
            "system": self.system,
            "raw_output": self.raw_output,
        }

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> "DocumentPrediction":
        return cls(
            doc_id=record["doc_id"],
            answers=[PredictedInconsistencyRecord.from_json(a) for a in record.get("answers", [])],
            system=record.get("system"),
            raw_output=record.get("raw_output"),
        )


def write_jsonl(records: Iterable[Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            payload = record.to_json() if hasattr(record, "to_json") else record
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_raw_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a local .parquet file into the same list[dict] shape read_raw_json produces."""
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def read_gold_jsonl(path: str | Path) -> dict[str, GoldExample]:
    examples = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = GoldExample.from_json(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({e})") from e

            examples[record.doc_id] = record
    return examples


def read_predictions_jsonl(path: str | Path) -> dict[str, DocumentPrediction]:
    predictions = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = DocumentPrediction.from_json(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({e})") from e

            predictions[record.doc_id] = record
    return predictions


def make_blind_predictions(gold_examples: Iterable[GoldExample], system: str) -> list[DocumentPrediction]:
    return [DocumentPrediction(doc_id=ex.doc_id, answers=[], system=system) for ex in gold_examples]
