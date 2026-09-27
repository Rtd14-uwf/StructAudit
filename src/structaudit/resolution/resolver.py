"""
Target resolution: resolves a StructuralCandidate's target_query to an actual unit_id.

Resolution order:
  1/2. exact + normalized structural identifier match (collapsed into one
       step here -- see note below)
  3. relative target ("following list" / "following table"): first
     compatible unit after the anchor, found via a plain text search for
     the anchor rather than a stored character offset
  4. local structural search (same-section, ranked by distance): NOT
     IMPLEMENTED.
  5. semantic fallback (local embeddings): NOT IMPLEMENTED.
"""

from __future__ import annotations

import re

from structaudit.candidates._shared import TARGET_GROUP
from structaudit.core import (
    ParsedDocument,
    ReferenceKind,
    ResolutionMethod,
    StructuralCandidate,
    StructuralIndexes,
    TargetResolution,
    UnitType,
)
from structaudit.indexing.indexer import normalize_identifier

_TARGET_GROUP_FULL_MATCH = re.compile(r"^" + TARGET_GROUP + r"$", re.IGNORECASE)

_KIND_TO_INDEX_FIELD = {
    ReferenceKind.TABLE: "table_number_to_unit",
    ReferenceKind.FIGURE: "figure_number_to_unit",
    ReferenceKind.CHART: "chart_number_to_unit",
    ReferenceKind.SECTION: "section_number_to_unit",
    ReferenceKind.CHAPTER: "chapter_number_to_unit",
    ReferenceKind.APPENDIX: "appendix_label_to_unit",
    ReferenceKind.NOTE: "note_number_to_unit",
    ReferenceKind.FOOTNOTE: "footnote_number_to_unit",
}

_RELATIVE_TARGETS = {"following list", "following table"}


def _resolve_explicit(target_query: str, indexes: StructuralIndexes) -> TargetResolution | None:
    """Steps 1/2: normalize the query's identifier and look it up directly."""
    match = _TARGET_GROUP_FULL_MATCH.match(target_query.strip())
    if not match:
        return None

    kind = ReferenceKind[match.group(1).upper()]
    identifier = match.group(2)
    key = normalize_identifier(kind, identifier)
    index_field = _KIND_TO_INDEX_FIELD[kind]
    unit_id = getattr(indexes, index_field).get(key)

    if unit_id is None:
        return TargetResolution(unit_id=None, method=ResolutionMethod.UNRESOLVED, confidence=0.0)

    return TargetResolution(unit_id=unit_id, method=ResolutionMethod.NORMALIZED, confidence=1.0)


def _resolve_relative(
    candidate: StructuralCandidate, parsed: ParsedDocument
) -> TargetResolution:
    """Step 3: "following list" / "following table" """
    anchor_position = parsed.raw_text.find(candidate.anchor_text)
    search_start = anchor_position + len(candidate.anchor_text) if anchor_position >= 0 else 0

    if candidate.target_query == "following table":
        candidates_in_order = [t for t in parsed.tables if (t.char_start or 0) >= search_start]
        candidates_in_order.sort(key=lambda t: t.char_start or 0)
        if candidates_in_order:
            return TargetResolution(
                unit_id=candidates_in_order[0].unit_id, method=ResolutionMethod.RELATIVE, confidence=0.8
            )
    else:  # "following list"
        lists_in_order = [
            u for u in parsed.units
            if u.unit_type == UnitType.LIST and (u.char_start or 0) >= search_start
        ]
        lists_in_order.sort(key=lambda u: u.char_start or 0)
        if lists_in_order:
            return TargetResolution(
                unit_id=lists_in_order[0].unit_id, method=ResolutionMethod.RELATIVE, confidence=0.8
            )

    return TargetResolution(unit_id=None, method=ResolutionMethod.UNRESOLVED, confidence=0.0)


def resolve_target(
    candidate: StructuralCandidate, parsed: ParsedDocument, indexes: StructuralIndexes
) -> TargetResolution:
    if candidate.target_query is None:
        return TargetResolution(unit_id=None, method=ResolutionMethod.UNRESOLVED, confidence=0.0)

    explicit_result = _resolve_explicit(candidate.target_query, indexes)
    if explicit_result is not None:
        return explicit_result

    if candidate.target_query in _RELATIVE_TARGETS:
        return _resolve_relative(candidate, parsed)

    return TargetResolution(unit_id=None, method=ResolutionMethod.UNRESOLVED, confidence=0.0)
