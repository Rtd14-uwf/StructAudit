"""
End-to-end pipeline: raw document text in, warnings out.
    parse -> structural indexes -> extract candidates -> rank
          -> resolve each candidate's target -> run each candidate's operations
          -> aggregate (policy, merge, evidence, budget)

audit_document() runs one document. warnings_to_prediction() converts the emitted warnings to the FIND prediction format
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from structaudit.aggregation.aggregator import AuditWarning, ValidatedCandidate, WarningPolicy, aggregate
from structaudit.candidates.extract import extract_candidates
from structaudit.core import (
    OperationType,
    ParsedDocument,
    StructuralCandidate,
    StructuralIndexes,
    TargetResolution,
    VerificationResult,
)
from structaudit.evaluation.schema import DocumentPrediction, PredictedInconsistencyRecord
from structaudit.indexing.indexer import build_structural_indexes
from structaudit.models.semantic_backend import LexicalOverlapBackend, SemanticBackend
from structaudit.parsing.parser import parse_document
from structaudit.ranking.ranker import rank_candidates
from structaudit.resolution.resolver import resolve_target
from structaudit.verifiers.count_validator import validate_count_match
from structaudit.verifiers.coverage_validator import validate_coverage_match
from structaudit.verifiers.reference_validator import (
    validate_target_exists,
    validate_target_label_match,
    validate_target_topic_match,
)
from structaudit.verifiers.schema_validator import validate_schema_header_match, validate_schema_row_match

SYSTEM_NAME = "structaudit"


@dataclass
class AuditResult:
    document_id: str
    warnings: list[AuditWarning]  # emitted: policy applied, budget applied
    held_back: list[AuditWarning]  # VIOLATED verdicts the policy did not emit
    n_candidates: int
    verdicts: Counter  # "OPERATION:STATUS:det|sem" -> count, for diagnostics
    errors: list[str] = field(default_factory=list)


def _run_operation(
    operation: OperationType,
    candidate: StructuralCandidate,
    resolution: TargetResolution,
    parsed: ParsedDocument,
    indexes: StructuralIndexes,
    backend: SemanticBackend,
    abbreviations: dict[str, str] | None,
) -> VerificationResult | None:
    if operation == OperationType.TARGET_EXISTS:
        return validate_target_exists(candidate, resolution, indexes)
    if operation == OperationType.TARGET_TOPIC_MATCH:
        return validate_target_topic_match(candidate, resolution, parsed, backend)
    if operation == OperationType.TARGET_LABEL_MATCH:
        return validate_target_label_match(candidate, resolution, parsed)
    if operation == OperationType.COUNT_MATCH:
        return validate_count_match(candidate, resolution, parsed)
    if operation == OperationType.COVERAGE_SET_MATCH:
        return validate_coverage_match(candidate, resolution, parsed, backend, abbreviations)
    if operation == OperationType.SCHEMA_HEADER_MATCH:
        return validate_schema_header_match(candidate, resolution, parsed, backend)
    if operation == OperationType.SCHEMA_ROW_MATCH:
        return validate_schema_row_match(candidate, resolution, parsed, backend)
    return None  # STRUCTURAL_ALIGNMENT: no candidate producer yet


def audit_document(
    text: str,
    document_id: str = "",
    backend: SemanticBackend | None = None,
    policy: WarningPolicy | None = None,
    abbreviations: dict[str, str] | None = None,
) -> AuditResult:
    backend = backend or LexicalOverlapBackend()
    policy = policy or WarningPolicy()

    parsed = parse_document(text)
    indexes = build_structural_indexes(parsed)
    candidates = rank_candidates(extract_candidates(parsed, document_id))

    validated: list[ValidatedCandidate] = []
    errors: list[str] = []
    for candidate in candidates:
        resolution = resolve_target(candidate, parsed, indexes)
        for operation in candidate.operations:
            try:
                result = _run_operation(operation, candidate, resolution, parsed, indexes, backend, abbreviations)
            except Exception as exc:  # one bad candidate must not sink the document
                errors.append(f"{candidate.candidate_id}/{operation.value}: {type(exc).__name__}: {exc}")
                continue
            if result is not None:
                validated.append(ValidatedCandidate(candidate, resolution, result))

    verdicts = Counter(
        f"{v.result.operations_run[0].value}:{v.result.status.value}:{'det' if v.result.deterministic else 'sem'}"
        for v in validated
    )
    warnings, held_back = aggregate(validated, text, document_id, policy)
    return AuditResult(document_id, warnings, held_back, len(candidates), verdicts, errors)


def warnings_to_prediction(document_id: str, warnings: list[AuditWarning]) -> DocumentPrediction:
    """FIND format: one answer per emitted warning (evidence + description)."""
    return DocumentPrediction(
        doc_id=document_id,
        answers=[PredictedInconsistencyRecord(evidence=list(w.evidence_text), description=w.explanation) for w in warnings],
        system=SYSTEM_NAME,
    )
