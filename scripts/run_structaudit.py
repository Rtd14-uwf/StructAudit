#!/usr/bin/env python3
"""
run_structaudit.py -- run the full StructAudit pipeline over a gold JSONL file and write FIND-format
predictions:

    python scripts/run_structaudit.py \
        --dataset data/processed/find/validation.jsonl \
        --output results/structaudit_validation.jsonl \
        [--warnings-output results/structaudit_warnings.jsonl] [--limit 20]

then score them:

    python scripts/evaluate_find.py --predictions results/structaudit_validation.jsonl \
        --gold data/processed/find/validation.jsonl --output results/metrics_validation.json

By default only deterministic VIOLATED verdicts become warnings. Violations that depend on the semantic
backend (a lexical stand-in) are held back and counted in the summary; --include-semantic emits them too,
for experiments. Every document in the dataset gets a prediction line (empty if nothing was flagged).
A document that crashes the pipeline gets an empty prediction and is reported in the summary.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from structaudit.aggregation.aggregator import WarningPolicy
from structaudit.core import OperationType
from structaudit.evaluation.schema import DocumentPrediction, read_gold_jsonl, write_jsonl
from structaudit.pipeline import SYSTEM_NAME, audit_document, warnings_to_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Gold jsonl (GoldExample schema)")
    parser.add_argument("--output", required=True, help="FIND-format predictions jsonl")
    parser.add_argument("--warnings-output", default=None, help="Optional: full structured warnings (emitted and held back), jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N documents")
    parser.add_argument("--max-warnings", type=int, default=5, help="Warning budget per document (default 5)")
    parser.add_argument("--include-semantic", action="store_true",
                        help="Also emit violations that depend on the semantic backend (off by default)")
    parser.add_argument("--allow-semantic", nargs="+", default=[], metavar="OPERATION",
                        help="Emit semantic-backend violations for just these operations, e.g. TARGET_TOPIC_MATCH")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Drop warnings below this confidence")
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        raise SystemExit(f"--dataset file not found: {args.dataset}")
    try:
        gold_examples = list(read_gold_jsonl(args.dataset).values())
    except ValueError as e:
        raise SystemExit(str(e))
    if args.limit:
        gold_examples = gold_examples[: args.limit]

    try:
        allowed = frozenset(OperationType[name.upper()] for name in args.allow_semantic)
    except KeyError as e:
        raise SystemExit(f"--allow-semantic: unknown operation {e}. Choose from: {', '.join(o.name for o in OperationType)}")

    policy = WarningPolicy(
        require_deterministic=not args.include_semantic,
        allow_semantic_for=allowed,
        min_confidence=args.min_confidence,
        max_warnings=args.max_warnings,
    )

    predictions: list[DocumentPrediction] = []
    warning_rows: list[dict] = []
    emitted_by_op: Counter = Counter()
    held_by_reason: Counter = Counter()
    docs_with_warning = 0
    crashed: list[str] = []
    internal_errors: list[str] = []
    started = time.time()

    for i, gold in enumerate(gold_examples, start=1):
        try:
            result = audit_document(gold.document_text, gold.doc_id, policy=policy)
        except Exception as exc:
            crashed.append(f"{gold.doc_id}: {type(exc).__name__}: {exc}")
            predictions.append(DocumentPrediction(doc_id=gold.doc_id, answers=[], system=SYSTEM_NAME))
            continue
        predictions.append(warnings_to_prediction(gold.doc_id, result.warnings))
        docs_with_warning += bool(result.warnings)
        internal_errors.extend(f"{gold.doc_id}: {e}" for e in result.errors)
        for w in result.warnings:
            emitted_by_op[w.operation.value] += 1
        for w in result.held_back:
            held_by_reason[f"{w.hold_reason} ({w.operation.value})"] += 1
        for w in result.warnings + result.held_back:
            warning_rows.append(json.loads(w.model_dump_json()))
        if i % 25 == 0:
            print(f"  processed {i}/{len(gold_examples)} documents...")

    write_jsonl(predictions, args.output)
    if args.warnings_output:
        write_jsonl(warning_rows, args.warnings_output)

    print(f"\n{len(gold_examples)} documents in {time.time() - started:.0f}s -> {args.output}")
    print(f"documents with at least one warning: {docs_with_warning}")
    print(f"warnings emitted:   {sum(emitted_by_op.values())}  {dict(emitted_by_op)}")
    print(f"violations held back by the policy: {sum(held_by_reason.values())}")
    for reason, n in held_by_reason.most_common():
        print(f"    {n:4}  {reason}")
    if internal_errors:
        print(f"validator errors (candidate skipped, document continued): {len(internal_errors)}")
        for e in internal_errors[:5]:
            print("   ", e)
    if crashed:
        print(f"documents that crashed the pipeline (empty prediction written): {len(crashed)}")
        for e in crashed[:5]:
            print("   ", e)
    print(f"\nnext: python scripts/evaluate_find.py --predictions {args.output} --gold {args.dataset} --output <metrics.json>")


if __name__ == "__main__":
    main()
