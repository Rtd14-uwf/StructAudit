"""
Core data types shared across the pipeline: parsed document structure
(DocumentUnit, TableUnit), extraction candidates (StructuralCandidate),
and validation results (VerificationResult), plus the enums and support
types the later modules build on.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class UnitType(str, Enum):
    DOCUMENT = "DOCUMENT"
    TOC_ENTRY = "TOC_ENTRY"
    SECTION = "SECTION"
    SUBSECTION = "SUBSECTION"
    PARAGRAPH = "PARAGRAPH"
    SENTENCE = "SENTENCE"
    LIST = "LIST"
    LIST_ITEM = "LIST_ITEM"
    TABLE = "TABLE"
    TABLE_ROW = "TABLE_ROW"
    TABLE_COLUMN = "TABLE_COLUMN"
    TABLE_CELL = "TABLE_CELL"
    FIGURE = "FIGURE"
    CAPTION = "CAPTION"
    FOOTNOTE = "FOOTNOTE"
    NOTE = "NOTE"
    APPENDIX = "APPENDIX"


class CandidateType(str, Enum):
    REFERENCE = "REFERENCE"
    COUNT = "COUNT"
    COVERAGE = "COVERAGE"
    SCHEMA = "SCHEMA"


class OperationType(str, Enum):
    TARGET_EXISTS = "TARGET_EXISTS"
    STRUCTURAL_ALIGNMENT = "STRUCTURAL_ALIGNMENT"
    TARGET_LABEL_MATCH = "TARGET_LABEL_MATCH"
    TARGET_TOPIC_MATCH = "TARGET_TOPIC_MATCH"
    COUNT_MATCH = "COUNT_MATCH"
    COVERAGE_SET_MATCH = "COVERAGE_SET_MATCH"
    SCHEMA_HEADER_MATCH = "SCHEMA_HEADER_MATCH"
    SCHEMA_ROW_MATCH = "SCHEMA_ROW_MATCH"


class Status(str, Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNCERTAIN = "UNCERTAIN"
    UNRESOLVED = "UNRESOLVED"


class ReferenceKind(str, Enum):
    """Kinds matched by the explicit structural reference regex, e.g. "Table 4", "footnote 49"."""

    TABLE = "TABLE"
    FIGURE = "FIGURE"
    CHART = "CHART"
    APPENDIX = "APPENDIX"
    SECTION = "SECTION"
    CHAPTER = "CHAPTER"
    NOTE = "NOTE"
    FOOTNOTE = "FOOTNOTE"


class DocumentUnit(BaseModel):
    unit_id: str
    unit_type: UnitType
    parent_id: str | None = None
    order: int = Field(ge=0)
    text: str
    label: str | None = None
    title: str | None = None
    caption: str | None = None
    section_path: list[str] = Field(default_factory=list)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    parse_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_char_span(self) -> "DocumentUnit":
        if self.char_start is not None and self.char_end is not None and self.char_end < self.char_start:
            raise ValueError(f"char_end < char_start for {self.unit_id!r}")
        return self

    @model_validator(mode="after")
    def _check_parent_not_self(self) -> "DocumentUnit":
        if self.parent_id is not None and self.parent_id == self.unit_id:
            raise ValueError(f"{self.unit_id!r} cannot be its own parent")
        return self


class TableUnit(BaseModel):
    """A table's row/column structure, kept separate from DocumentUnit so it isn't lost by flattening."""

    unit_id: str
    order: int = Field(ge=0)
    label: str | None = None
    caption: str | None = None
    column_headers: list[str] = Field(default_factory=list)
    row_headers: list[str] = Field(default_factory=list)
    cells: list[list[str]] = Field(default_factory=list)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    parse_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_char_span(self) -> "TableUnit":
        if self.char_start is not None and self.char_end is not None and self.char_end < self.char_start:
            raise ValueError(f"char_end < char_start for {self.unit_id!r}")
        return self

    @model_validator(mode="after")
    def _check_cells_rectangular(self) -> "TableUnit":
        """Row-count validators downstream assume 'cells' is a proper grid"""
        if self.cells:
            row_lengths = {len(row) for row in self.cells}
            if len(row_lengths) > 1:
                raise ValueError(f"cells not rectangular for {self.unit_id!r}: row lengths {sorted(row_lengths)}")
        return self


class StructuralCandidate(BaseModel):
    candidate_id: str
    document_id: str
    anchor_unit_id: str
    anchor_text: str
    candidate_type: CandidateType
    target_query: str | None = None
    target_kind: str | None = None
    expected_members: list[str] = Field(default_factory=list)
    expected_count: int | None = Field(default=None, ge=0)
    expected_schema: list[str] = Field(default_factory=list)
    expected_topic: str | None = None
    operations: list[OperationType] = Field(default_factory=list)
    rule_id: str
    rule_confidence: float = Field(ge=0.0, le=1.0)
    parse_confidence: float = Field(ge=0.0, le=1.0)
    rank_score: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _check_type_specific_required_fields(self) -> "StructuralCandidate":
        """Each candidate_type must carry the field it exists to check"""
        if self.candidate_type == CandidateType.COUNT and self.expected_count is None:
            raise ValueError("COUNT candidate missing expected_count")

        if self.candidate_type == CandidateType.COVERAGE and not self.expected_members:
            raise ValueError("COVERAGE candidate missing expected_members")

        if self.candidate_type == CandidateType.SCHEMA and not self.expected_schema:
            raise ValueError("SCHEMA candidate missing expected_schema")

        if self.candidate_type == CandidateType.REFERENCE and not self.target_query:
            raise ValueError("REFERENCE candidate missing target_query")

        return self


class VerificationResult(BaseModel):
    candidate_id: str
    status: Status
    operations_run: list[OperationType]
    expected: dict
    observed: dict
    evidence_unit_ids: list[str]
    evidence_text: list[str]
    deterministic: bool
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str

    @model_validator(mode="after")
    def _check_unresolved_has_no_evidence(self) -> "VerificationResult":
        if self.status == Status.UNRESOLVED and self.evidence_unit_ids:
            raise ValueError(f"{self.candidate_id!r} is UNRESOLVED but carries evidence_unit_ids")
        return self


class ReferenceMention(BaseModel):
    """One explicit structural pointer found in raw text ("Table 4", "footnote 49"), prior to resolution or validation."""
    raw_text: str
    kind: ReferenceKind
    identifier_raw: str  # as it appeared, unnormalized: "03", "V", "b", "049"
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_char_span(self) -> "ReferenceMention":
        if self.char_end < self.char_start:
            raise ValueError("char_end < char_start")
        return self


class ParsedDocument(BaseModel):
    document_id: str | None = None
    raw_text: str
    units: list[DocumentUnit] = Field(default_factory=list)
    tables: list[TableUnit] = Field(default_factory=list)
    reference_mentions: list[ReferenceMention] = Field(default_factory=list)


class StructuralIndexes(BaseModel):
    """Deterministic lookup indexes for retrieving structural objects by normalized identifier."""
    table_number_to_unit: dict[str, str] = Field(default_factory=dict)
    figure_number_to_unit: dict[str, str] = Field(default_factory=dict)
    chart_number_to_unit: dict[str, str] = Field(default_factory=dict)
    section_number_to_unit: dict[str, str] = Field(default_factory=dict)
    chapter_number_to_unit: dict[str, str] = Field(default_factory=dict)
    appendix_label_to_unit: dict[str, str] = Field(default_factory=dict)
    note_number_to_unit: dict[str, str] = Field(default_factory=dict)
    footnote_number_to_unit: dict[str, str] = Field(default_factory=dict)
    toc_entries: list[str] = Field(default_factory=list)
    headings_by_order: list[str] = Field(default_factory=list)
    tables_by_order: list[str] = Field(default_factory=list)
    figures_by_order: list[str] = Field(default_factory=list)
    lists_by_order: list[str] = Field(default_factory=list)


class ExtractionRule(BaseModel):
    """Metadata describing one candidate-extraction rule"""
    rule_id: str
    candidate_type: CandidateType
    trigger_kind: str
    operations: list[OperationType] = Field(default_factory=list)
    base_confidence: float = Field(ge=0.0, le=1.0)


class ResolutionMethod(str, Enum):
    """How a candidate's target_query was resolved to a unit_id, per the
    priority order target resolution follows: exact/normalized identifier
    match first, then relative, then local search, then semantic fallback.
    """

    NORMALIZED = "NORMALIZED"  # exact and normalized identifier match collapse into one step (see resolver.py)
    RELATIVE = "RELATIVE"
    LOCAL_SEARCH = "LOCAL_SEARCH"
    SEMANTIC = "SEMANTIC"
    UNRESOLVED = "UNRESOLVED"


class TargetResolution(BaseModel):
    """Result of resolving one candidate's target_query to a unit."""
    unit_id: str | None
    method: ResolutionMethod
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_unresolved_has_no_unit(self) -> "TargetResolution":
        if self.method == ResolutionMethod.UNRESOLVED and self.unit_id is not None:
            raise ValueError("UNRESOLVED resolution cannot carry a unit_id")
        if self.method != ResolutionMethod.UNRESOLVED and self.unit_id is None:
            raise ValueError(f"{self.method.value} resolution must carry a unit_id")
        return self


class SemanticCallLog(BaseModel):
    task: str
    model_backend: str  # "local" | "api"
    model_name: str
    input: dict
    raw_output: dict
    parsed_output: dict
    token_usage: int | None = None
    latency: float = Field(ge=0.0)
