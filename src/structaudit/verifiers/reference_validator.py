"""
REFERENCE Validator (all four checks): TARGET_EXISTS, STRUCTURAL_ALIGNMENT,
TARGET_LABEL_MATCH, TARGET_TOPIC_MATCH.

TARGET_EXISTS is purely deterministic (structural index lookup).
STRUCTURAL_ALIGNMENT, TARGET_LABEL_MATCH, and TARGET_TOPIC_MATCH all
compare text and fall back to a SemanticBackend (see
models/semantic_backend.py) for the fuzzy middle ground.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from structaudit.core import (
    OperationType,
    ParsedDocument,
    Status,
    StructuralCandidate,
    TargetResolution,
    VerificationResult,
)
from structaudit.models.semantic_backend import SemanticBackend

# Section 11.2's own thresholds, verbatim.
_ALIGNMENT_SATISFIED_THRESHOLD = 95
_ALIGNMENT_VIOLATED_THRESHOLD = 70


def _normalize_for_alignment(text: str) -> str:
    """Case, whitespace, punctuation, and numbering normalization"""
    text = text.lower().strip()
    text = re.sub(r"^(chapter|section|appendix)\s+", "", text)  # numbering prefix
    text = re.sub(r"[^\w\s]", "", text)  # punctuation
    text = re.sub(r"\s+", " ", text)  # whitespace
    return text.strip()


def validate_target_exists(
    candidate: StructuralCandidate, resolution: TargetResolution
) -> VerificationResult:
    if resolution.unit_id is None:
        return VerificationResult(
            candidate_id=candidate.candidate_id,
            status=Status.VIOLATED,
            operations_run=[OperationType.TARGET_EXISTS],
            expected={"target_query": candidate.target_query},
            observed={"resolved": False},
            evidence_unit_ids=[],
            evidence_text=[],
            deterministic=True,
            confidence=1.0,
            explanation=f"{candidate.target_query!r} does not resolve to any unit in the document.",
        )

    return VerificationResult(
        candidate_id=candidate.candidate_id,
        status=Status.SATISFIED,
        operations_run=[OperationType.TARGET_EXISTS],
        expected={"target_query": candidate.target_query},
        observed={"resolved": True, "unit_id": resolution.unit_id},
        evidence_unit_ids=[resolution.unit_id],
        evidence_text=[],
        deterministic=True,
        confidence=1.0,
        explanation=f"{candidate.target_query!r} resolved to {resolution.unit_id!r}.",
    )


def validate_structural_alignment(
    candidate_id: str,
    source_text: str,
    target_text: str,
    backend: SemanticBackend,
    evidence_unit_ids: list[str] | None = None,
) -> VerificationResult:

    source_norm = _normalize_for_alignment(source_text)
    target_norm = _normalize_for_alignment(target_text)
    similarity = fuzz.token_sort_ratio(source_norm, target_norm)

    if similarity >= _ALIGNMENT_SATISFIED_THRESHOLD:
        return VerificationResult(
            candidate_id=candidate_id, status=Status.SATISFIED,
            operations_run=[OperationType.STRUCTURAL_ALIGNMENT],
            expected={"source_text": source_text}, observed={"target_text": target_text, "similarity": similarity},
            evidence_unit_ids=evidence_unit_ids or [], evidence_text=[target_text],
            deterministic=True, confidence=1.0,
            explanation=f"{source_text!r} and {target_text!r} align (similarity={similarity:.0f}).",
        )
    if similarity < _ALIGNMENT_VIOLATED_THRESHOLD:
        return VerificationResult(
            candidate_id=candidate_id, status=Status.VIOLATED,
            operations_run=[OperationType.STRUCTURAL_ALIGNMENT],
            expected={"source_text": source_text}, observed={"target_text": target_text, "similarity": similarity},
            evidence_unit_ids=evidence_unit_ids or [], evidence_text=[target_text],
            deterministic=True, confidence=1.0,
            explanation=f"{source_text!r} and {target_text!r} do not align (similarity={similarity:.0f}).",
        )

    # 70 <= similarity < 95: ambiguous, defer to the semantic backend.
    semantic_result = backend.classify("reference_topic_support", {
        "expected_topic": source_text, "resolved_target_text": target_text,
    })
    label = semantic_result["label"]
    status = {"SUPPORTED": Status.SATISFIED, "NOT_SUPPORTED": Status.VIOLATED, "UNCERTAIN": Status.UNCERTAIN}[label]
    return VerificationResult(
        candidate_id=candidate_id, status=status,
        operations_run=[OperationType.STRUCTURAL_ALIGNMENT],
        expected={"source_text": source_text},
        observed={"target_text": target_text, "similarity": similarity, "semantic_label": label},
        evidence_unit_ids=evidence_unit_ids or [], evidence_text=[target_text],
        deterministic=False, confidence=semantic_result["score"] / 100,
        explanation=f"Similarity {similarity:.0f} was ambiguous; semantic check returned {label}.",
    )


def validate_target_label_match(
    candidate: StructuralCandidate, resolution: TargetResolution, parsed: ParsedDocument
) -> VerificationResult:
    if resolution.unit_id is None or candidate.expected_topic is None:
        return VerificationResult(
            candidate_id=candidate.candidate_id, status=Status.UNRESOLVED,
            operations_run=[OperationType.TARGET_LABEL_MATCH],
            expected={"expected_identity": candidate.expected_topic}, observed={},
            evidence_unit_ids=[], evidence_text=[],
            deterministic=True, confidence=1.0,
            explanation="No resolved target or no expected identity to compare.",
        )

    target_identity = _find_unit_identity(resolution.unit_id, parsed)
    if target_identity is None:
        return VerificationResult(
            candidate_id=candidate.candidate_id, status=Status.UNCERTAIN,
            operations_run=[OperationType.TARGET_LABEL_MATCH],
            expected={"expected_identity": candidate.expected_topic}, observed={"unit_id": resolution.unit_id},
            evidence_unit_ids=[resolution.unit_id], evidence_text=[],
            deterministic=False, confidence=0.0,
            explanation=f"{resolution.unit_id!r} has no title/caption/label to compare against.",
        )

    similarity = fuzz.token_set_ratio(
        _normalize_for_alignment(candidate.expected_topic), _normalize_for_alignment(target_identity)
    )

    status = Status.SATISFIED if similarity >= _ALIGNMENT_SATISFIED_THRESHOLD else Status.VIOLATED
    return VerificationResult(
        candidate_id=candidate.candidate_id, status=status,
        operations_run=[OperationType.TARGET_LABEL_MATCH],
        expected={"expected_identity": candidate.expected_topic},
        observed={"target_identity": target_identity, "similarity": similarity},
        evidence_unit_ids=[resolution.unit_id], evidence_text=[target_identity],
        deterministic=True, confidence=1.0,
        explanation=f"Expected identity {candidate.expected_topic!r} vs target {target_identity!r} (similarity={similarity:.0f}).",
    )


def _find_unit_identity(unit_id: str, parsed: ParsedDocument) -> str | None:
    for table in parsed.tables:
        if table.unit_id == unit_id:
            return table.caption or table.label
    for unit in parsed.units:
        if unit.unit_id == unit_id:
            return unit.title or unit.caption or unit.label
    return None


def validate_target_topic_match(
    candidate: StructuralCandidate, resolution: TargetResolution, parsed: ParsedDocument, backend: SemanticBackend
) -> VerificationResult:
    if resolution.unit_id is None or candidate.expected_topic is None:
        return VerificationResult(
            candidate_id=candidate.candidate_id, status=Status.UNRESOLVED,
            operations_run=[OperationType.TARGET_TOPIC_MATCH],
            expected={"expected_topic": candidate.expected_topic}, observed={},
            evidence_unit_ids=[], evidence_text=[],
            deterministic=True, confidence=1.0,
            explanation="No resolved target or no expected topic to check.",
        )

    resolved_target_text = _find_unit_text(resolution.unit_id, parsed)
    semantic_result = backend.classify("reference_topic_support", {
        "expected_topic": candidate.expected_topic, "resolved_target_text": resolved_target_text or "",
    })
    label = semantic_result["label"]
    status = {"SUPPORTED": Status.SATISFIED, "NOT_SUPPORTED": Status.VIOLATED, "UNCERTAIN": Status.UNCERTAIN}[label]

    return VerificationResult(
        candidate_id=candidate.candidate_id, status=status,
        operations_run=[OperationType.TARGET_TOPIC_MATCH],
        expected={"expected_topic": candidate.expected_topic},
        observed={"resolved_target_text": resolved_target_text, "semantic_label": label, "score": semantic_result["score"]},
        evidence_unit_ids=[resolution.unit_id], evidence_text=[resolved_target_text] if resolved_target_text else [],
        deterministic=False, confidence=semantic_result["score"] / 100,
        explanation=f"Topic {candidate.expected_topic!r} vs target text: semantic check returned {label}.",
    )


def _find_unit_text(unit_id: str, parsed: ParsedDocument) -> str | None:
    for table in parsed.tables:
        if table.unit_id == unit_id:
            parts = [table.caption or "", " ".join(table.column_headers)]
            parts.extend(" ".join(row) for row in table.cells[:3])  # a few rows, not the whole table
            return " ".join(p for p in parts if p).strip() or None
    for unit in parsed.units:
        if unit.unit_id == unit_id:
            return unit.text or unit.title
    return None
