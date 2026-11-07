"""Aggregation and preregistered gate for the shrinkage experiment."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


METHODS: tuple[str, ...] = (
    "nearest_source",
    "equal_ensemble",
    "similarity_top1",
    "similarity_weighted",
    "similarity_shrunk",
)
METRICS: tuple[str, ...] = (
    "rmse",
    "mae",
    "kge",
    "nse",
    "peak_level_error_m",
    "peak_time_error_h",
)
RMSE_THRESHOLD_M = 0.2027
WIN_RATE_THRESHOLD = 0.5


def validate_seed_metric_grid(
    frame: pd.DataFrame,
    *,
    target_ids: Sequence[str],
    event_ids: Sequence[str],
    seeds: Sequence[int],
) -> None:
    required = {
        "target_station_id",
        "event_id",
        "seed",
        "method",
        *METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"seed metrics missing columns: {missing}")
    key_columns = ["target_station_id", "event_id", "seed", "method"]
    if frame.duplicated(key_columns).any():
        raise ValueError("seed metrics contain duplicate keys")
    expected = {
        (str(target), str(event), int(seed), method)
        for target in target_ids
        for event in event_ids
        for seed in seeds
        for method in METHODS
    }
    actual = {
        (
            str(row.target_station_id),
            str(row.event_id),
            int(row.seed),
            str(row.method),
        )
        for row in frame.itertuples(index=False)
    }
    if actual != expected:
        raise ValueError(
            "seed metrics must contain the exact shrinkage grid "
            f"(missing={len(expected - actual)}, extra={len(actual - expected)})"
        )
    core = frame[
        ["rmse", "mae", "peak_level_error_m", "peak_time_error_h"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(core.to_numpy(float)).all():
        raise ValueError("core shrinkage metrics must be finite")


def aggregate_seed_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    grouping = ["target_station_id", "event_id", "method"]
    working = pd.concat(
        [
            frame[grouping].reset_index(drop=True),
            frame[list(METRICS)].apply(pd.to_numeric, errors="coerce"),
        ],
        axis=1,
    )
    aggregate = working.groupby(grouping, sort=False, observed=True)[
        list(METRICS)
    ].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate["method"] = pd.Categorical(
        aggregate["method"], categories=list(METHODS), ordered=True
    )
    return aggregate.sort_values(
        ["target_station_id", "event_id", "method"]
    ).reset_index(drop=True)


def build_overall_summary(task_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        subset = task_metrics.loc[task_metrics["method"] == method]
        if subset.empty:
            raise ValueError(f"task metrics missing method: {method}")
        row: dict[str, str | int | float] = {
            "method": method,
            "task_count": int(len(subset)),
        }
        for metric in METRICS:
            values = pd.to_numeric(subset[f"{metric}_mean"], errors="coerce")
            row[f"{metric}_task_mean"] = float(values.mean())
            row[f"{metric}_task_median"] = float(values.median())
            row[f"{metric}_task_std"] = float(values.std())
        rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise_win_rates(task_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["target_station_id", "event_id"]
    proposed = task_metrics.loc[
        task_metrics["method"] == "similarity_shrunk"
    ].set_index(keys)
    rows = []
    for baseline_method in ("equal_ensemble", "similarity_weighted"):
        baseline = task_metrics.loc[
            task_metrics["method"] == baseline_method
        ].set_index(keys)
        if not proposed.index.equals(baseline.index):
            raise ValueError(f"task keys do not match {baseline_method}")
        row: dict[str, str | int | float] = {
            "proposed_method": "similarity_shrunk",
            "baseline_method": baseline_method,
            "task_count": int(len(proposed)),
        }
        for metric in ("rmse", "mae"):
            proposed_values = proposed[f"{metric}_mean"].to_numpy(float)
            baseline_values = baseline[f"{metric}_mean"].to_numpy(float)
            row[f"{metric}_win_rate"] = float(
                np.mean(proposed_values < baseline_values)
            )
            row[f"{metric}_tie_rate"] = float(
                np.mean(np.isclose(proposed_values, baseline_values))
            )
        for metric in ("kge", "nse"):
            proposed_values = proposed[f"{metric}_mean"].to_numpy(float)
            baseline_values = baseline[f"{metric}_mean"].to_numpy(float)
            finite = np.isfinite(proposed_values) & np.isfinite(baseline_values)
            row[f"{metric}_win_rate"] = (
                float(np.mean(proposed_values[finite] > baseline_values[finite]))
                if finite.any()
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_success_gate(mean_rmse: float, equal_win_rate: float) -> dict:
    rmse_passed = float(mean_rmse) < RMSE_THRESHOLD_M
    win_rate_passed = float(equal_win_rate) > WIN_RATE_THRESHOLD
    return {
        "mean_rmse_m": float(mean_rmse),
        "equal_rmse_win_rate": float(equal_win_rate),
        "rmse_threshold_m": RMSE_THRESHOLD_M,
        "win_rate_threshold": WIN_RATE_THRESHOLD,
        "rmse_passed": bool(rmse_passed),
        "win_rate_passed": bool(win_rate_passed),
        "passed": bool(rmse_passed and win_rate_passed),
    }
