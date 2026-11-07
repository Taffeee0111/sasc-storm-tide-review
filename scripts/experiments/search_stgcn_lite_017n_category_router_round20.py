"""Target-blind category router (round 20) -- production finalization path.



This rewrite makes the whole path label-free:

* Diagnostics are computed only from the frozen, label-free quantities:
  the Stage-1 ``similarity_shrunk`` base prediction, the graph-branch
  output, its anchor-hull clip, and the anchor consensus.
* The whitelist is selected by a fixed diagnostic score applied to the
  base-vs-graph peak disagreement itself; no observed value is consulted.
* Router parameters are selected per active target by
  leave-one-pseudo-target-out validation over the 42 pseudo-target tasks
  (six non-target stations x seven non-test events).  Label reads during
  prediction go exclusively through
  :class:`src.target_blind_router.PseudoTargetLabelStore`, which rejects
  any key (s, v) with s == active station or v == active event.
* Predictions are written to disk and sealed with a SHA-256 manifest
  *before* the active target's record is read; the post-seal evaluator is
  the only component that reads it.

Link to the reported conclusions
--------------------------------
This script is the single entry point that produces the artifacts behind
the manuscript's two-stage-method numbers:

* ``<result_dir>/predictions/*.csv``      -- the sealed per-task final
  predictions whose ``predicted_water_level_m`` column is the method's
  output for all 56 (target station, storm) tasks;
* ``<result_dir>/metrics/per_task_metrics.csv`` -- per-task RMSE/MAE/KGE/
  NSE/PBIAS/RSR/Willmott-d and peak errors used for the per-station and
  per-event tables and the failure-case figures;
* ``<result_dir>/metrics/summary.csv``    -- the mean-task summary row
  (mean RMSE, peak-time/height errors, POD/FAR/CSI) that feeds the main
  comparison table;
* ``<result_dir>/predictions_evaluated/*.csv`` -- the sealed predictions
  joined with ``observed_water_level_m``, written by the evaluator after
  sealing for figure generation.

All of them are produced by this one script run, in the order prescribed
by the protocol: finalize -> seal -> evaluate.

Usage
-----
From the repository root::

    python scripts/experiments/search_stgcn_lite_017n_category_router_round20.py

"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = _SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.target_blind_router import (  # noqa: E402
    DEFAULT_PEAK_THRESHOLD_M,
    FILE_PATTERN,
    PROTOCOL_VERSION,
    PseudoTargetLabelStore,
    TaskKey,
    build_diagnostics,
    build_whitelist,
    discover_tasks,
    finalize_task_predictions,
    load_truth_frame,
    seal_predictions,
    verify_seal,
)
from src.metrics import evaluate_event  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "results" / "backbone_followups" / "017n_category_router_round20"
RESULT_DIR = OUTPUT_ROOT / "stgcn_lite_017n_cr_r20_tb_c0001"

# Detection threshold used by the post-seal evaluator.  This is the same

# the POD/FAR/CSI reporting.
PEAK_THRESHOLD = DEFAULT_PEAK_THRESHOLD_M


def compute_event_metrics(
    observed: np.ndarray, predicted: np.ndarray, timestamps: pd.Series
) -> dict[str, float]:
    """
    Only ever called in the post-seal evaluation phase.
    """
    metrics = evaluate_event(observed, predicted, timestamps)
    observed_exceeds = observed >= PEAK_THRESHOLD
    predicted_exceeds = predicted >= PEAK_THRESHOLD
    hits = int(np.sum(observed_exceeds & predicted_exceeds))
    misses = int(np.sum(observed_exceeds & (~predicted_exceeds)))
    false_alarms = int(np.sum((~observed_exceeds) & predicted_exceeds))
    pod = np.nan if hits + misses == 0 else hits / (hits + misses)
    far = np.nan if hits + false_alarms == 0 else false_alarms / (hits + false_alarms)
    csi = np.nan if hits + misses + false_alarms == 0 else hits / (hits + misses + false_alarms)
    return {
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "kge": float(metrics["kge"]),
        "nse": float(metrics["nse"]),
        "pbias": float(metrics["pbias"]),
        "rsr": float(metrics["rsr"]),
        "willmott_d": float(metrics["willmott_d"]),
        "kge_r": float(metrics["kge_r"]),
        "kge_alpha": float(metrics["kge_alpha"]),
        "kge_beta": float(metrics["kge_beta"]),
        "peak_level_error_m": float(metrics["peak_level_error_m"]),
        "peak_time_error_h": float(metrics["peak_time_error_h"]),
        "peak_level_error_abs": abs(float(metrics["peak_level_error_m"])),
        "peak_time_error_abs": abs(float(metrics["peak_time_error_h"])),
        "peak_level_error_abs_true": abs(float(metrics["peak_level_error_m"])),
        "peak_time_error_abs_true": abs(float(metrics["peak_time_error_h"])),
        "pod": float(pod) if pd.notna(pod) else np.nan,
        "far": float(far) if pd.notna(far) else np.nan,
        "csi": float(csi) if pd.notna(csi) else np.nan,
        "global_peak_threshold_m": PEAK_THRESHOLD,
    }


def finalize(truth_frame: pd.DataFrame, tasks: list[TaskKey]) -> dict:
    """Prediction phase.

    Builds the label-free diagnostics, selects the whitelist and the
    per-task router parameters from pseudo-targets only, writes the 56
    sealed prediction files, and returns the router records.  The active
    target's record is never read here: all label access goes through
    :class:`PseudoTargetLabelStore`, whose forbidden set is
    ``{(s, v): s == active station or v == active event}``.
    """
    (RESULT_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    diags = build_diagnostics(tasks)
    whitelist = build_whitelist(diags)
    label_store = PseudoTargetLabelStore(truth_frame)

    records: list[dict] = []
    for active in tasks:
        frame, _, _, record = finalize_task_predictions(
            diags, active, label_store, whitelist
        )
        frame["graph_argmax_changed"] = int(
            int(np.argmax(frame["predicted_water_level_m"].to_numpy(float)))
            != int(np.argmax(frame["base_predicted_water_level_m"].to_numpy(float)))
        )
        frame.to_csv(
            RESULT_DIR
            / "predictions"
            / f"target_{active.station}__event_{active.event}__seed_20260723.csv",
            index=False,
        )
        records.append(record)

    seal_path = seal_predictions(RESULT_DIR, records)
    return {
        "seal_path": str(seal_path),
        "task_count": len(tasks),
        "whitelist": sorted(whitelist),
        "records": records,
    }


def evaluate(finalize_info: dict) -> None:
    """Evaluation phase -- the only place the active target's record is read.

    Verifies the seal first; computes per-task and aggregate metrics;
    writes the legacy-format ``metrics/summary.csv``, a per-task table, the
    evaluated prediction copies used for figures, and the run manifest.
    """
    manifest = verify_seal(RESULT_DIR)
    truth_frame = load_truth_frame()

    rows: list[dict] = []
    evaluated_dir = RESULT_DIR / "predictions_evaluated"
    evaluated_dir.mkdir(exist_ok=True)

    for pred_path in sorted((RESULT_DIR / "predictions").glob("*.csv")):
        match = FILE_PATTERN.match(pred_path.name)
        if not match:
            continue
        station = str(match.group("station"))
        event = str(match.group("event"))
        seed = int(match.group("seed"))
        frame = pd.read_csv(pred_path, parse_dates=["timestamp_utc"])

        truth = truth_frame.loc[
            (truth_frame["station_id"] == station) & (truth_frame["event_id"] == event),
            ["timestamp_utc", "water_level_m"],
        ].sort_values("timestamp_utc")
        merged = frame.merge(truth, on="timestamp_utc", how="inner")
        if len(merged) != len(frame):
            raise ValueError(f"truth alignment failed for {station}::{event}")

        metrics = compute_event_metrics(
            merged["water_level_m"].to_numpy(float),
            merged["predicted_water_level_m"].to_numpy(float),
            merged["timestamp_utc"],
        )
        rows.append(
            {
                "target_station_id": station,
                "event_id": event,
                "seed": seed,
                "subgroup": str(frame["router_subgroup"].iloc[0]),
                "router_trigger_mean": float(frame["router_trigger_mean"].iloc[0]),
                "whitelist_hit": int(frame["whitelist_hit"].iloc[0]),
                "graph_argmax_changed": int(frame["graph_argmax_changed"].iloc[0]),
                "delta_timing_h": float(frame["delta_timing_h"].iloc[0]),
                "delta_height_m": float(frame["delta_height_m"].iloc[0]),
                **metrics,
            }
        )

        evaluated = frame.copy()
        evaluated["observed_water_level_m"] = merged["water_level_m"].to_numpy(float)
        evaluated.to_csv(evaluated_dir / pred_path.name, index=False)

    per_task = pd.DataFrame(rows).sort_values(["target_station_id", "event_id"])
    per_task.to_csv(RESULT_DIR / "metrics" / "per_task_metrics.csv", index=False)

    # Mean-task summary row.  Metric names and aggregation order match the
    # legacy summary.csv so the same downstream tables can be rebuilt.
    mean_summary = per_task[
        [
            "rmse",
            "mae",
            "kge",
            "nse",
            "pbias",
            "rsr",
            "willmott_d",
            "kge_r",
            "kge_alpha",
            "kge_beta",
            "peak_level_error_m",
            "peak_time_error_h",
            "pod",
            "far",
            "csi",
            "global_peak_threshold_m",
            "peak_level_error_abs_true",
            "peak_time_error_abs_true",
        ]
    ].mean(numeric_only=True)
    summary: dict[str, float | int | str] = {key: float(value) for key, value in mean_summary.to_dict().items()}
    summary["task_count"] = int(len(per_task))
    summary["whitelist_count"] = int(per_task["whitelist_hit"].sum())
    summary["whitelist_changed_count"] = int((per_task["whitelist_hit"] & per_task["graph_argmax_changed"]).sum())
    summary["graph_relocate_count"] = int(per_task["subgroup"].isin(
        ["whitelist", "graph_timing_strong", "graph_timing_soft"]
    ).sum())
    summary["graph_argmax_changed_count"] = int(per_task["graph_argmax_changed"].sum())
    summary["pbias_abs"] = abs(summary["pbias"])
    summary["kge_alpha_dist"] = abs(summary["kge_alpha"] - 1.0)
    summary["kge_beta_dist"] = abs(summary["kge_beta"] - 1.0)
    summary["peak_level_error_abs"] = abs(summary["peak_level_error_m"])
    summary["peak_time_error_abs"] = abs(summary["peak_time_error_h"])
    summary["validation_mean_nrmse"] = float(
        np.mean([record["validation_mean_nrmse"] for record in finalize_info["records"]])
    )
    summary["result_dir"] = RESULT_DIR.name

    pd.DataFrame([summary]).to_csv(RESULT_DIR / "metrics" / "summary.csv", index=False)

    run_manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "target_blind": True,
        "result_dir": RESULT_DIR.name,
        "task_count": int(len(per_task)),
        "whitelist_tasks": sorted(finalize_info["whitelist"]),
        "seal_protocol_version": manifest.get("protocol_version"),
        "mean_task_rmse": float(summary["rmse"]),
        "mean_task_mae": float(summary["mae"]),
        "mean_abs_peak_level_error_m": float(per_task["peak_level_error_abs_true"].mean()),
        "mean_abs_peak_time_error_h": float(per_task["peak_time_error_abs_true"].mean()),
        "mean_pod": float(summary["pod"]),
        "mean_far": float(summary["far"]),
        "mean_csi": float(summary["csi"]),
        "router_records": finalize_info["records"],
    }
    (RESULT_DIR / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"TASK_COUNT={int(len(per_task))}")
    print(f"WHITELIST_TASKS={sorted(finalize_info['whitelist'])}")
    print(f"MEAN_TASK_RMSE={float(summary['rmse']):.6f}")
    print(f"MEAN_ABS_PEAK_TIME_ERROR_H={float(per_task['peak_time_error_abs_true'].mean()):.6f}")
    print(f"MEAN_POD={float(summary['pod']):.6f}")
    print(f"MEAN_FAR={float(summary['far']):.6f}")
    print(f"MEAN_CSI={float(summary['csi']):.6f}")


def main() -> None:
    started = datetime.now(timezone.utc)
    truth_frame = load_truth_frame()
    tasks = discover_tasks()
    finalize_info = finalize(truth_frame, tasks)
    evaluate(finalize_info)
    print(f"FINISHED_UTC={datetime.now(timezone.utc).isoformat()}")
    print(f"STARTED_UTC={started.isoformat()}")


if __name__ == "__main__":
    main()
