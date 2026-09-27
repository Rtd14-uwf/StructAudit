#!/usr/bin/env python3
"""
evaluate_find.py

Compute FIND response-level metrics for a predictions file against a gold
file:

    python scripts/evaluate_find.py \
        --predictions results/structaudit_validation.jsonl \
        --gold data/processed/find/validation.jsonl \
        --output results/metrics_validation.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from structaudit.evaluation.find_metrics import (
    evidence_metric_strict,
    response_level_score,
)
from structaudit.evaluation.schema import (
    DocumentPrediction,
    GoldExample,
    read_gold_jsonl,
    read_predictions_jsonl,
)


def per_document_evidence_score(gold: GoldExample, prediction: DocumentPrediction | None) -> float:
    """Response-level evidence score for one document: best-matching candidate's F1."""
    if prediction is None or not prediction.answers:
        result = evidence_metric_strict([], gold.evidence, gold.document_text)
        return result.f1

    candidate_scores = []
    for answer in prediction.answers:
        result = evidence_metric_strict(answer.evidence, gold.evidence, gold.document_text)
        candidate_scores.append(result.f1)
    return response_level_score(candidate_scores)


@dataclass
class EvidenceScoreSummary:
    by_source: dict[str, float]
    avg: float
    n_documents: int


@dataclass
class NotImplementedMetric:
    reason: str
    not_implemented: bool = True


@dataclass
class FindEvaluationResult:
    evidence_score: EvidenceScoreSummary
    description_score: NotImplementedMetric
    task_score: NotImplementedMetric
    n_missing_predictions: int
    missing_prediction_doc_ids: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


def evaluate(
    gold_examples: dict[str, GoldExample],
    predictions: dict[str, DocumentPrediction],
) -> FindEvaluationResult:
    """Compute per-source and overall evidence score. A gold doc_id missing
    from predictions counts as a miss rather than being skipped.
    """
    scores_by_source: dict[str, list[float]] = {}
    missing_predictions = []

    for doc_id, gold in gold_examples.items():
        prediction = predictions.get(doc_id)
        if prediction is None:
            missing_predictions.append(doc_id)
        score = per_document_evidence_score(gold, prediction)
        scores_by_source.setdefault(gold.source, []).append(score)

    per_source_mean = {
        source: round(statistics.mean(scores), 4) for source, scores in scores_by_source.items()
    }
    all_scores = [s for scores in scores_by_source.values() for s in scores]
    overall_mean = round(statistics.mean(all_scores), 4) if all_scores else 0.0

    return FindEvaluationResult(
        evidence_score=EvidenceScoreSummary(
            by_source=per_source_mean,
            avg=overall_mean,
            n_documents=len(all_scores),
        ),
        description_score=NotImplementedMetric(reason="BLEURT backend not wired up yet"),
        task_score=NotImplementedMetric(reason="LLM-as-judge backend not wired up yet"),
        n_missing_predictions=len(missing_predictions),
        missing_prediction_doc_ids=missing_predictions[:20],  # sample, not the full list
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Path to predictions .jsonl")
    parser.add_argument("--gold", required=True, help="Path to gold .jsonl")
    parser.add_argument("--output", required=True, help="Path to write metrics .json")
    args = parser.parse_args()

    for label, path in (("--gold", args.gold), ("--predictions", args.predictions)):
        if not Path(path).exists():
            raise SystemExit(f"{label} file not found: {path}")

    try:
        gold_examples = read_gold_jsonl(args.gold)
        predictions = read_predictions_jsonl(args.predictions)
    except ValueError as e:
        raise SystemExit(str(e))

    if not gold_examples:
        raise SystemExit(f"No gold examples loaded from {args.gold}")

    metrics = evaluate(gold_examples, predictions)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics.to_json(), indent=2), encoding="utf-8")

    print(f"Wrote metrics for {len(gold_examples)} gold documents to {output_path}")
    print(json.dumps(asdict(metrics.evidence_score), indent=2))


if __name__ == "__main__":
    main()
