"""Strict aggregation helpers for the exhaustive closed-book experiment."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


METHODS: tuple[str, ...] = (
    "nearest_source",
    "equal_ensemble",
    "similarity_top1",
    "similarity_weighted",
)
METRICS: tuple[str, ...] = (
    "rmse",
    "mae",
    "kge",
    "nse",
    "peak_level_error_m",
    "peak_time_error_h",
)
KEY_COLUMNS: tuple[str, ...] = (
    "target_station_id",
    "event_id",
    "seed",
    "method",
)


def validate_seed_metric_grid(
    frame: pd.DataFrame,
    *,
    target_ids: Sequence[str],
    event_ids: Sequence[str],
    seeds: Sequence[int],
) -> None:
    """Reject incomplete, duplicated, or unexpected evaluation grids."""
    required = {*KEY_COLUMNS, *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"seed metric table is missing columns: {missing}")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("seed metric table contains duplicate task-method keys")
    expected = {
        (str(target), str(event), int(seed), method)
        for target in target_ids
        for event in event_ids
        for seed in seeds
        for method in METHODS
    }
    actual = {
        (str(row.target_station_id), str(row.event_id), int(row.seed), row.method)
        for row in frame.itertuples(index=False)
    }
    if actual != expected:
        missing_keys = len(expected - actual)
        extra_keys = len(actual - expected)
        raise ValueError(
            "seed metric table must contain the exact frozen grid "
            f"(missing={missing_keys}, extra={extra_keys})"
        )
    values = frame[list(METRICS)].apply(pd.to_numeric, errors="coerce")
    if values[["rmse", "mae", "peak_level_error_m", "peak_time_error_h"]].isna().any().any():
        raise ValueError("core metrics must be numeric and non-missing")


def aggregate_seed_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the three seeds within each target-event-method task."""
    grouping = ["target_station_id", "event_id", "method"]
    numeric = frame[list(METRICS)].apply(pd.to_numeric, errors="coerce")
    working = pd.concat([frame[grouping].reset_index(drop=True), numeric], axis=1)
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
    """Summarize task-level means so every target-event receives equal weight."""
    rows = []
    for method in METHODS:
        subset = task_metrics.loc[task_metrics["method"] == method]
        if subset.empty:
            raise ValueError(f"task metrics are missing method {method}")
        row: dict[str, float | int | str] = {
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
    """Compare the proposed weighted method with each fixed baseline."""
    proposed = task_metrics.loc[
        task_metrics["method"] == "similarity_weighted"
    ].set_index(["target_station_id", "event_id"])
    if proposed.index.has_duplicates:
        raise ValueError("proposed task metrics contain duplicate keys")
    rows = []
    for baseline_method in METHODS:
        if baseline_method == "similarity_weighted":
            continue
        baseline = task_metrics.loc[
            task_metrics["method"] == baseline_method
        ].set_index(["target_station_id", "event_id"])
        if not proposed.index.equals(baseline.index):
            raise ValueError(
                f"task keys for {baseline_method} do not match proposed method"
            )
        row: dict[str, float | int | str] = {
            "proposed_method": "similarity_weighted",
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
