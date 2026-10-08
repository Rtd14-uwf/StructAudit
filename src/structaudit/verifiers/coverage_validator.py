"""
COVERAGE Validator: checks that every member a COVERAGE candidate says is
"shown/listed/included in" a target appears in that target.

Observed content is built from the complete resolved target:
  - table:   every header, row label, and cell, across ALL fragments that
             share the table's label
  - list:    every item, including items of nested lists
  - section / paragraph / note / footnote: the full text (for a section, the
             body up to the next heading at the same or a higher level)

Each expected member is matched based on the following order:
  1. normalized exact   (whole element equal)
  2. token-normalized   (order-insensitive / plural equality)
  3. abbreviation       (the same comparison after expanding abbreviations)
  4. semantic           (SemanticBackend, over the most similar observed items)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process, utils

from structaudit.core import (
    DocumentUnit,
    OperationType,
    ParsedDocument,
    Status,
    StructuralCandidate,
    TableUnit,
    TargetResolution,
    UnitType,
    VerificationResult,
)
from structaudit.models.semantic_backend import SemanticBackend
from structaudit.verifiers._matching import fold_plural, normalize_exact

_SEMANTIC_TASK = "coverage_member_equivalence"
_SEMANTIC_CANDIDATE_LIMIT = 10
_MAX_REPORTED_ELEMENTS = 20

_HEADING_TYPES = {UnitType.SECTION, UnitType.SUBSECTION, UnitType.APPENDIX}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

DEFAULT_ABBREVIATIONS: dict[str, str] = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april",
    "jun": "june", "jul": "july", "aug": "august", "sep": "september",
    "sept": "september", "oct": "october", "nov": "november", "dec": "december",
    "avg": "average", "approx": "approximately", "qtr": "quarter", "yr": "year",
    "pct": "percent", "govt": "government", "mfg": "manufacturing",
    "dept": "department", "intl": "international", "incl": "including",
    "excl": "excluding", "yoy": "year over year", "qoq": "quarter over quarter",
}


@dataclass(frozen=True)
class _Text:
    """One piece of text that is tokenized in three ways."""
    text: str
    tokens: tuple[str, ...]  # normalized
    folded: tuple[str, ...]  # + plural folding
    canon: tuple[str, ...]  # + abbreviation expansion


def _normalize_abbreviations(abbreviations: dict[str, str]) -> dict[str, tuple[str, ...]]:
    normalized = {}
    for short, long_form in abbreviations.items():
        key = normalize_exact(short)
        if key and " " not in key:
            normalized[key] = tuple(normalize_exact(long_form).split())
    return normalized


def _make_text(text: str, abbreviations: dict[str, tuple[str, ...]]) -> _Text:
    tokens = tuple(normalize_exact(text).split())
    folded = tuple(fold_plural(t) for t in tokens)
    canon = tuple(fold_plural(part) for t in tokens for part in abbreviations.get(t, (t,)))
    return _Text(text=text, tokens=tokens, folded=folded, canon=canon)


def _contains_run(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    n = len(needle)
    return n > 0 and any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def _containment_is_trustworthy(tokens: tuple[str, ...]) -> bool:
    joined = "".join(tokens)
    return len(joined) >= 3 or (any(c.isdigit() for c in joined) and any(c.isalpha() for c in joined))


def _find_target(unit_id: str, parsed: ParsedDocument) -> TableUnit | DocumentUnit | None:
    for table in parsed.tables:
        if table.unit_id == unit_id:
            return table
    return next((u for u in parsed.units if u.unit_id == unit_id), None)


def _list_item_texts(list_unit: DocumentUnit, parsed: ParsedDocument) -> list[str]:
    list_ids = {list_unit.unit_id}
    texts = []
    # nested lists have the outer list as parent_id; their items have the nested list's id
    changed = True
    while changed:
        changed = False
        for u in parsed.units:
            if u.unit_type == UnitType.LIST and u.parent_id in list_ids and u.unit_id not in list_ids:
                list_ids.add(u.unit_id)
                changed = True
    for u in parsed.units:
        if u.unit_type == UnitType.LIST_ITEM and u.parent_id in list_ids and u.text:
            texts.append(u.text)
    return texts


def _section_text(unit: DocumentUnit, parsed: ParsedDocument) -> str:
    """Heading title plus the body up to the next heading at the same or a higher level."""
    if unit.char_start is None or unit.char_end is None:
        return unit.text
    depth = len(unit.section_path)
    ends = [
        other.char_start for other in parsed.units
        if other.unit_type in _HEADING_TYPES and other.char_start is not None
        and other.char_start > unit.char_start and len(other.section_path) <= depth
    ]
    end = min(ends, default=len(parsed.raw_text))
    return unit.text + "\n" + parsed.raw_text[unit.char_end:end]


def _observed(
    target: TableUnit | DocumentUnit, parsed: ParsedDocument
) -> tuple[list[str], bool, list[str]]:
    """Returns (observed texts, is_prose, unit_ids inspected)."""
    if isinstance(target, TableUnit):
        fragments = [t for t in parsed.tables if target.label and t.label == target.label] or [target]
        texts: list[str] = []
        for table in fragments:
            texts.extend(table.column_headers)
            texts.extend(table.row_headers)
            texts.extend(cell for row in table.cells for cell in row)
        return [t for t in dict.fromkeys(texts) if t.strip()], False, [t.unit_id for t in fragments]

    if target.unit_type == UnitType.LIST:
        return list(dict.fromkeys(_list_item_texts(target, parsed))), False, [target.unit_id]

    text = _section_text(target, parsed) if target.unit_type in _HEADING_TYPES else target.text
    return [text] if text.strip() else [], True, [target.unit_id]


def _stage_exact(member: _Text, elements: list[_Text], is_prose: bool):
    trustworthy = _containment_is_trustworthy(member.tokens)
    for e in elements:
        if member.tokens == e.tokens or (is_prose and trustworthy and _contains_run(e.tokens, member.tokens)):
            return "normalized_exact", e
    return None


def _stage_folded(member: _Text, elements: list[_Text], attr: str, method: str):
    member_tokens = getattr(member, attr)
    for e in elements:
        if sorted(member_tokens) == sorted(getattr(e, attr)):
            return method, e
    if _containment_is_trustworthy(member.tokens):
        for e in elements:
            if _contains_run(getattr(e, attr), member_tokens):
                return ("token_contained" if method == "token_normalized" else method), e
    return None


def _semantic_candidates(member: str, texts: list[str], is_prose: bool) -> list[str]:
    pool = texts
    if is_prose:
        pool = [s.strip() for t in texts for s in _SENTENCE_SPLIT_RE.split(t) if s.strip()]
    pool = list(dict.fromkeys(pool))
    ranked = process.extract(
        member, pool, scorer=fuzz.token_sort_ratio, processor=utils.default_process, limit=_SEMANTIC_CANDIDATE_LIMIT
    )
    return [choice for choice, _score, _index in ranked]


def validate_coverage_match(
    candidate: StructuralCandidate,
    resolution: TargetResolution,
    parsed: ParsedDocument,
    backend: SemanticBackend,
    abbreviations: dict[str, str] | None = None,
) -> VerificationResult:
    expected = candidate.expected_members
    operation = OperationType.COVERAGE_SET_MATCH

    if resolution.unit_id is None:
        return VerificationResult(
            candidate_id=candidate.candidate_id, status=Status.UNRESOLVED,
            operations_run=[operation],
            expected={"expected_members": expected}, observed={},
            evidence_unit_ids=[], evidence_text=[],
            deterministic=True, confidence=1.0,
            explanation="No resolved target to check coverage of.",
        )

    target = _find_target(resolution.unit_id, parsed)
    if target is None:
        return VerificationResult(
            candidate_id=candidate.candidate_id, status=Status.UNCERTAIN,
            operations_run=[operation],
            expected={"expected_members": expected}, observed={"unit_id": resolution.unit_id},
            evidence_unit_ids=[resolution.unit_id], evidence_text=[],
            deterministic=False, confidence=0.0,
            explanation=f"Resolved unit {resolution.unit_id!r} not found in document.",
        )

    abbrev = _normalize_abbreviations(DEFAULT_ABBREVIATIONS if abbreviations is None else abbreviations)
    observed_texts, is_prose, inspected_ids = _observed(target, parsed)
    elements = [_make_text(t, abbrev) for t in observed_texts]

    matched: dict[str, str] = {}
    methods: dict[str, str] = {}
    missing: list[str] = []
    uncertain: list[str] = []
    semantic_calls = 0
    semantic_confidences: list[float] = []

    for member_text in expected:
        member = _make_text(member_text, abbrev)
        if not member.tokens:
            uncertain.append(member_text)  # nothing searchable (e.g. punctuation only)
            continue

        hit = (
            _stage_exact(member, elements, is_prose)
            or _stage_folded(member, elements, "folded", "token_normalized")
            or _stage_folded(member, elements, "canon", "abbreviation")
        )
        if hit:
            methods[member_text], element = hit
            matched[member_text] = element.text
            continue

        if not _containment_is_trustworthy(member.tokens) and any(_contains_run(e.folded, member.folded) for e in elements):
            uncertain.append(member_text)  # appears only inside longer text; too short to trust either way
            continue

        if not elements:
            missing.append(member_text)  # complete target inspected and it is empty
            continue

        semantic_calls += 1
        verdict = backend.classify(_SEMANTIC_TASK, {
            "expected_label": member_text,
            "candidate_labels": _semantic_candidates(member_text, observed_texts, is_prose),
        })
        if verdict["label"] == "SUPPORTED":
            methods[member_text] = "semantic"
            matched[member_text] = verdict["best_match"]
            semantic_confidences.append(verdict["confidence"])
        elif verdict["label"] == "NOT_SUPPORTED":
            missing.append(member_text)
            semantic_confidences.append(verdict["confidence"])
        else:
            uncertain.append(member_text)

    if missing:
        status = Status.VIOLATED
    elif uncertain:
        status = Status.UNCERTAIN
    else:
        status = Status.SATISFIED

    deterministic = semantic_calls == 0 and status != Status.UNCERTAIN
    confidence = 0.0 if status == Status.UNCERTAIN else min(semantic_confidences, default=1.0)

    if status == Status.SATISFIED:
        explanation = f"All {len(expected)} expected members found."
    elif status == Status.VIOLATED:
        explanation = f"Not found in the target: {missing}."
    else:
        explanation = f"Could not decide whether {uncertain} appear in the target."

    observed = {
        "matched_members": list(matched),
        "missing_members": missing,
        "match_methods": methods,
        "observed_member_count": len(observed_texts),
        "observed_members": observed_texts[:_MAX_REPORTED_ELEMENTS] if not is_prose else [],
        "observed_members_truncated": len(observed_texts) > _MAX_REPORTED_ELEMENTS and not is_prose,
    }
    if uncertain:
        observed["uncertain_members"] = uncertain

    evidence = [] if is_prose else list(dict.fromkeys(matched.values()))[:_MAX_REPORTED_ELEMENTS]

    return VerificationResult(
        candidate_id=candidate.candidate_id, status=status,
        operations_run=[operation],
        expected={"expected_members": expected},
        observed=observed,
        evidence_unit_ids=inspected_ids, evidence_text=evidence,
        deterministic=deterministic, confidence=confidence,
        explanation=explanation,
    )
