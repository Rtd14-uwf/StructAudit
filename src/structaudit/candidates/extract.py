"""
Candidate extractor entry point: runs all four rule families (reference,
count, coverage, schema) and combines their output. Each family numbers
its own candidate_ids with a distinct prefix, so there's no cross-family collision
"""

from __future__ import annotations

from structaudit.candidates.count_rules import extract_count_candidates
from structaudit.candidates.coverage_rules import extract_coverage_candidates
from structaudit.candidates.reference_rules import extract_reference_candidates
from structaudit.candidates.schema_rules import extract_schema_candidates
from structaudit.core import ParsedDocument, StructuralCandidate


def extract_candidates(parsed: ParsedDocument, document_id: str | None = None) -> list[StructuralCandidate]:
    """Run all four rule families and return every candidate found, with
    document_id set on each at construction time.
    """
    text = parsed.raw_text
    doc_id = document_id or ""
    candidates: list[StructuralCandidate] = []
    candidates.extend(extract_reference_candidates(text, parsed.reference_mentions, doc_id))
    candidates.extend(extract_count_candidates(text, doc_id))
    candidates.extend(extract_coverage_candidates(text, doc_id))
    candidates.extend(extract_schema_candidates(text, doc_id))
    return candidates
