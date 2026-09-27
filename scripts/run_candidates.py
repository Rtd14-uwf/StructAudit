#!/usr/bin/env python3
"""
run_candidates.py

Candidate extraction evaluation:

    python scripts/run_candidates.py \
        --dataset data/processed/find/validation.jsonl \
        --top-k 5 \
        --output results/candidates_validation.jsonl

For each gold document: parse -> extract candidates -> rank -> check
whether the gold evidence is surfaced within the top-1/3/5/all ranked
candidates (see candidate_eval.py for what "surfaced" means). Writes the
full ranked candidate list per document to --output and prints aggregate
Recall@k and volume metrics to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from structaudit.candidates.extract import extract_candidates
from structaudit.evaluation.candidate_eval import aggregate_candidate_eval, evaluate_document
from structaudit.evaluation.schema import read_gold_jsonl
from structaudit.parsing.parser import parse_document
from structaudit.ranking.ranker import rank_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Gold jsonl (GoldExample schema)")
    parser.add_argument("--top-k", type=int, default=5, help="Primary warning budget (informational; recall is reported at 1/3/5/all regardless)")
    parser.add_argument("--output", required=True, help="Per-document ranked candidates, jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N documents (for a quick smoke run)")
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        raise SystemExit(f"--dataset file not found: {args.dataset}")

    try:
        gold_examples = list(read_gold_jsonl(args.dataset).values())
    except ValueError as e:
        raise SystemExit(str(e))
    if args.limit:
        gold_examples = gold_examples[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eval_results = []
    with output_path.open("w", encoding="utf-8") as out_f:
        for i, gold in enumerate(gold_examples):
            parsed = parse_document(gold.document_text)
            candidates = extract_candidates(parsed, document_id=gold.doc_id)
            ranked = rank_candidates(candidates)

            eval_results.append(evaluate_document(gold.doc_id, ranked, gold.evidence))

            out_f.write(json.dumps({
                "doc_id": gold.doc_id,
                "n_candidates": len(ranked),
                "candidates": [
                    {
                        "candidate_type": c.candidate_type.value,
                        "target_query": c.target_query,
                        "anchor_text": c.anchor_text,
                        "rank_score": round(c.rank_score, 4),
                    }
                    for c in ranked
                ],
            }) + "\n")

            if (i + 1) % 25 == 0:
                print(f"  processed {i + 1}/{len(gold_examples)} documents...")

    metrics = aggregate_candidate_eval(eval_results)
    print()
    print(json.dumps(metrics.to_json(), indent=2))

    # Documents that surfaced nothing at all -- the starting point for
    # inspecting misses.
    misses = [r.doc_id for r in eval_results if not r.surfaced_at_all]
    print()
    print(f"{len(misses)} / {len(eval_results)} documents surfaced nothing (Recall@All miss)")


if __name__ == "__main__":
    main()
