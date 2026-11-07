# SASC storm-tide study: source review materials

Supporting source materials for **Guarded Graph Correction for Closed-Book Cross-Station Storm-Tide Prediction**.

**Release status: source-only review snapshot, prepared 7 November 2026. Syntax and a local path defect have been repaired, but this is not a complete, execution-verified reproduction release.** The code supplied here has not been executed or independently validated as the code that generated the manuscript's numerical results.

## Contents

- `src/` and `scripts/`: 37 Python source files, including the target-blind round20 router candidate and selected upstream reference dependencies.
- `configs/experiments/seven_station_56.yaml`: the directly referenced experiment configuration.
- `DEPENDENCIES_NOT_LOCKED.txt`: dependency names inferred from static imports; versions are unverified.
- `SOURCE_MANIFEST_SHA256.csv`: original source selection and hashes.
- `PUBLIC_SOURCE_MANIFEST_SHA256.csv`: current public file hashes and the two edited source files.
- `ORIGINAL_STATIC_INSPECTION.json`: historical inspection of the original source snapshot, before the repairs.
- `PUBLICATION_AUDIT.json`: static checks of this repaired public copy.
- `CHANGES.md`: exact scope and verification limits of the repairs.
- `DATA_AVAILABILITY.md`: required inputs, available local inventory, and the reasons the data are not distributed in this release.

## Repairs in this copy

1. Restored the missing end of the round20 module docstring and the opening triple quote of the `compute_event_metrics` docstring. These malformed quotes caused the original line-96 `IndentationError`.
2. Replaced the STGCN reference's machine-specific root path with `Path(__file__).resolve().parents[2]`, locating the repository root from the script.

No router rule, model computation, metric formula, experiment parameter, threshold, seed, or result has been changed. The original files remain separately preserved; this public copy is not represented as the historical executed revision.

## Remaining limitations

1. Required data, checkpoints, Stage-1 predictions, and graph predictions are not included. The exact source revision that generated the historical 017i graph inputs has not been established.
2. `load_truth_frame()` reads the full event table before prediction finalization. Guarded label access does not demonstrate physical isolation of every target observation; the complete upstream information boundary remains to be verified.
3. Historical outputs must not be relabeled as results produced by the rewritten target-blind candidate. This release does not establish the correspondence between the supplied code, upstream forecasts, and manuscript results.
4. Python/dependency versions are not locked. Static parsing, compilation, and import inspection do not demonstrate correct execution or numerical reproducibility.

Source comments that describe a final or production path remain historical source text, not certification of this release.

## Inspection performed

All 37 Python files passed AST parsing and in-memory compilation. Local import paths resolved within this copy. These checks did not execute the compiled code, import project modules, install dependencies, run project tests, train models, run inference, evaluate performance, or recompute manuscript metrics.

A usable reproduction release will additionally require the actual executed source revision, verified data and input provenance, a resolved environment, and a documented mapping to reported results. See `DATA_AVAILABILITY.md`.

## Rights and reuse

Public visibility provides access for inspection. No software or data reuse license is granted by this snapshot; a license has not yet been selected or verified. Third-party datasets and materials remain subject to their providers' terms.

Manuscript drafts, internal review notes, personal files, credentials, model weights, caches, and historical result directories are excluded.
