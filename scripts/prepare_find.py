#!/usr/bin/env python3
"""
prepare_find.py

Normalize raw FIND rows into the processed GoldExample schema. Accepts
either a local .parquet file downloaded from the HF Hub or a raw JSONL
file (auto-detected by extension):

    python scripts/prepare_find.py \
        --input data/test-00000-of-00001.parquet \
        --output data/processed/find/test.jsonl \
        --split test

Two ways to point this at different columns, if needed:

  1. --inspect: print the raw columns (and value types, for parquet) of
     the first row and exit.

  2. --field-map path/to/map.json: a JSON file overriding a subset of
     schema.py's _DEFAULT_FIELD_MAP, e.g. {"doc_id": ["uid"]}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from structaudit.evaluation.schema import (
    _DEFAULT_FIELD_MAP,
    hf_dataset_to_gold_examples,
    read_raw_parquet_rows,
    write_jsonl,
)


def read_raw_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_raw_rows(path: str | Path) -> list[dict]:
    """Dispatch on file extension: .parquet -> pandas, everything else -> JSONL."""
    path = Path(path)
    if path.suffix == ".parquet":
        return read_raw_parquet_rows(path)
    return read_raw_jsonl(path)


def load_field_map_override(path: str | Path | None) -> dict[str, tuple[str, ...]]:
    """Merge a JSON field-map override on top of the default; the override
    file only needs the keys that differ, e.g. {"doc_id": ["uid"]}.
    """
    merged = {k: tuple(v) for k, v in _DEFAULT_FIELD_MAP.items()}
    if path is None:
        return merged
    with Path(path).open("r", encoding="utf-8") as f:
        override = json.load(f)
    unknown_keys = set(override) - set(merged)
    if unknown_keys:
        raise ValueError(f"--field-map has unrecognized keys {sorted(unknown_keys)}, expected subset of {sorted(merged)}")
    for key, candidates in override.items():
        merged[key] = tuple(candidates)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Raw FIND rows, one JSON object per line")
    parser.add_argument("--output", help="Where to write the processed gold .jsonl (required unless --inspect)")
    parser.add_argument(
        "--split",
        choices=["test", "validation"],
        help="Split label to stamp onto every record (required unless --inspect)",
    )
    parser.add_argument(
        "--field-map",
        help="Optional JSON file overriding column-name guesses in schema.py's _DEFAULT_FIELD_MAP",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print the first row's columns and exit, without writing any output",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise SystemExit(f"Input file not found: {args.input}")

    rows = read_raw_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows read from {args.input}")

    if args.inspect:
        print(f"{len(rows)} rows read from {args.input}")
        print(f"Columns in first row: {sorted(rows[0].keys())}")
        if Path(args.input).suffix == ".parquet":
            print("Value types (nested columns often come back as numpy.ndarray):")
            print(json.dumps({k: type(v).__name__ for k, v in rows[0].items()}, indent=2))
        print("First row values (truncated):")
        for key, value in rows[0].items():
            preview = repr(list(value)) if isinstance(value, np.ndarray) else repr(value)
            if len(preview) > 300:
                preview = preview[:300] + "... (truncated)"
            print(f"  {key}: {preview}")
        print("Current default field map (schema.py._DEFAULT_FIELD_MAP):")
        print(json.dumps({k: list(v) for k, v in _DEFAULT_FIELD_MAP.items()}, indent=2))
        return

    if not args.output or not args.split:
        raise SystemExit("--output and --split are required unless --inspect is set")

    field_map = load_field_map_override(args.field_map)

    try:
        gold_examples = hf_dataset_to_gold_examples(rows, split=args.split, field_map=field_map)
    except KeyError as e:
        raise SystemExit(f"{e}\n\nRe-run with --inspect, then pass --field-map with a JSON override.")

    write_jsonl(gold_examples, args.output)
    print(f"Wrote {len(gold_examples)} gold examples to {args.output}")


if __name__ == "__main__":
    main()
