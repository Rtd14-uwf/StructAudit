# StructAudit

Typed structural consistency checking for long documents: finds places where a
document points at, counts, or describes something (a table, a footnote, a
list of items) that doesn't actually match what's there.

## Requirements

- Python 3.11+

## Installation

From the project root:

```bash
pip install -e .
```

This installs the `structaudit` package and its dependencies (pydantic,
pandas, markdown-it-py, and a few others).

## Running the tests

```bash
pytest tests/
```

You should see all tests pass. Run this after installing to confirm your
environment is set up correctly.

## Using it as a library

The core pipeline is: parse a document, extract candidates, rank them.

```python
from structaudit.parsing.parser import parse_document
from structaudit.candidates.extract import extract_candidates
from structaudit.ranking.ranker import rank_candidates

text = "See Table 3 for hours worked. BERT, T5, and RoBERTa are shown in Table 4."

parsed = parse_document(text)
candidates = extract_candidates(parsed, document_id="doc1")
ranked = rank_candidates(candidates)

for c in ranked:
    print(c.candidate_type.value, c.target_query, round(c.rank_score, 3))
```

```
COVERAGE Table 4 0.94
REFERENCE Table 3 0.93
REFERENCE Table 4 0.93
```

## Working with the FIND benchmark

The included scripts prepare and evaluate against the
[kensho/FIND](https://huggingface.co/datasets/kensho/FIND) benchmark. Run
them from the project root in this order.

### 1. Get the data
Download kensho/FIND `.parquet` files locally.

### 2. Normalize it into the gold schema

```bash
python scripts/prepare_find.py \
  --input data/raw/find/find_validation.parquet \
  --output data/processed/find/validation.jsonl \
  --split validation
```

If the column names don't match what's expected, run with `--inspect`
first to see what's actually in the file:

```bash
python scripts/prepare_find.py --input data/raw/find/find_validation.parquet --inspect
```

### 3. Run candidate extraction and see how well it does

```bash
python scripts/run_candidates.py \
  --dataset data/processed/find/validation.jsonl \
  --output results/candidates_validation.jsonl
```

This parses every document, extracts candidates, ranks them, and reports
Recall@1/3/5/All: how often the benchmark's known inconsistency is
surfaced within the top-ranked candidates.

```
{
  "n_documents": 125,
  "recall_at_1": 0.048,
  "recall_at_3": 0.096,
  "recall_at_5": 0.16,
  "recall_at_all": 0.2,
  ...
}
```

### 4. Evaluate a system's predictions against gold

If you have a system's predictions in the expected format (see
`DocumentPrediction` in `src/structaudit/evaluation/schema.py`), score them
against the gold evidence:

```bash
python scripts/evaluate_find.py \
  --predictions results/my_predictions.jsonl \
  --gold data/processed/find/validation.jsonl \
  --output results/metrics.json
```
