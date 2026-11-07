# Changes to the public source copy

Prepared 7 November 2026 with authorization to repair clear syntax and minor code defects. Original source files were not edited.

## 1. Round20 docstring delimiters

File: `scripts/experiments/search_stgcn_lite_017n_category_router_round20.py`

- Added the missing closing triple quote immediately after the module usage example, before `from __future__ import annotations`.
- Changed the single opening quote of the `compute_event_metrics` docstring into a triple quote.

The original source failed with `IndentationError: unexpected indent` at line 96. With the two delimiter repairs, the entry parses and compiles in memory. No executable statement, formula, threshold, seed, or algorithmic branch was edited.

## 2. Portable repository root in the STGCN reference

File: `scripts/experiments/run_stgcn_lite_seven_station.py`

Replaced a machine-specific Windows root path with:

```python
ROOT = Path(__file__).resolve().parents[2]
```

This derives the root from the existing `scripts/experiments/` layout, removes a personal home-directory path, and avoids the interim redaction placeholder. It does not supply missing data or establish that this reference generated the historical 017i forecasts.

## Provenance

`SOURCE_MANIFEST_SHA256.csv` retains original-source hashes. `PUBLIC_SOURCE_MANIFEST_SHA256.csv` records the current bytes and hashes: 36 unchanged source/configuration files and 2 edited source files.

`ORIGINAL_STATIC_INSPECTION.json` deliberately retains the pre-repair failure as historical evidence. `PUBLICATION_AUDIT.json` describes the repaired copy.

## Verification boundary

All 37 Python files passed AST parsing and in-memory compilation; local import paths were checked statically. No compiled code or project module was executed. No dependencies, training, inference, tests, evaluation, or metric recomputation were run. Syntax repair is not evidence that the original experiments ran under this revision or that numerical results reproduce.
