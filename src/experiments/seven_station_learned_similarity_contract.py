"""Meta-training contract for the seven-station source-prior learned
similarity metric (parallel to five_station_learned_similarity_contract.py,
pointed at seven_station_contract). The dynamic storm-shape feature from
Round 4 is intentionally omitted: it carried ~0 normalized importance and
~0 raw correlation with transfer_nrmse in the four-station panel."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from src.experiments.exhaustive_contract import EVENTS, SEEDS
from src.experiments.seven_station_contract import STATIONS

FEATURE_COLUMNS = (
    "atmosphere_distance",
    "wave_distance",
    "coastal_distance",
    "source_prior_error",
)
ALPHA_GRID = (0.0, 0.01, 0.1, 1.0, 10.0)
TEMPERATURE_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
META_COLUMNS = (
    "source_station_id",
    "pseudo_target_station_id",
    "event_id",
    "seed",
    *FEATURE_COLUMNS,
    "transfer_nrmse",
)

__all__ = [
    "FEATURE_COLUMNS",
    "ALPHA_GRID",
    "TEMPERATURE_GRID",
    "LAMBDA_GRID",
    "META_COLUMNS",
    "build_allowed_meta_grid",
    "attach_source_quality_prior",
    "attach_source_prior_error_std",
    "validate_meta_training_frame",
]


def build_allowed_meta_grid(active_target: str, test_event: str) -> pd.DataFrame:
    allowed_stations = [station for station in STATIONS if station != active_target]
    rows = []
    for pseudo_target in allowed_stations:
        for source in allowed_stations:
            if source == pseudo_target:
                continue
            for event in EVENTS:
                if event == test_event:
                    continue
                for seed in SEEDS:
                    rows.append(
                        {
                            "source_station_id": source,
                            "pseudo_target_station_id": pseudo_target,
                            "event_id": event,
                            "seed": seed,
                        }
                    )
    return pd.DataFrame(rows)


def attach_source_quality_prior(frame: pd.DataFrame) -> pd.DataFrame:
    """Add each row's source's leave-one-pseudo-target-out mean transfer error.

    For a row with source S and pseudo-target T, the prior excludes every
    row of the frame where source == S and pseudo_target == T, so no
    information about how S transfers to T ever informs S's own prior for
    that T. This only reuses transfer_nrmse values already present in the
    supplied (target-blind) frame; it reads no new data.
    """
    required = {"source_station_id", "pseudo_target_station_id", "transfer_nrmse"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns for source prior: {sorted(missing)}")

    result = frame.copy()
    source = result["source_station_id"].astype(str)
    pseudo_target = result["pseudo_target_station_id"].astype(str)

    source_totals = frame.groupby(frame["source_station_id"].astype(str))[
        "transfer_nrmse"
    ].sum()
    source_counts = frame.groupby(frame["source_station_id"].astype(str))[
        "transfer_nrmse"
    ].count()
    pair_totals = frame.groupby(
        [source, pseudo_target]
    )["transfer_nrmse"].sum()
    pair_counts = frame.groupby(
        [source, pseudo_target]
    )["transfer_nrmse"].count()

    total_for_source = source.map(source_totals).to_numpy(float)
    count_for_source = source.map(source_counts).to_numpy(float)
    pair_index = pd.MultiIndex.from_arrays([source, pseudo_target])
    own_pair_total = pair_totals.reindex(pair_index).to_numpy(float)
    own_pair_count = pair_counts.reindex(pair_index).to_numpy(float)

    remaining_total = total_for_source - own_pair_total
    remaining_count = count_for_source - own_pair_count
    if (remaining_count <= 0).any():
        raise ValueError(
            "source quality prior requires at least one other pseudo-target "
            "per source"
        )
    result["source_prior_error"] = remaining_total / remaining_count
    return result


def attach_source_prior_error_std(frame: pd.DataFrame) -> pd.DataFrame:
    """Add each row's source's leave-one-pseudo-target-out std of transfer
    error, using the same held-out-pair exclusion as
    attach_source_quality_prior. Captures how consistent a source's transfer
    performance has been across other pseudo-targets; used as a capability-
    aware shrinkage signal, not as a regression feature. Reuses only
    transfer_nrmse values already present in the supplied (target-blind)
    frame; it reads no new data.
    """
    required = {"source_station_id", "pseudo_target_station_id", "transfer_nrmse"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns for source prior std: {sorted(missing)}")

    result = frame.copy()
    source = result["source_station_id"].astype(str)
    pseudo_target = result["pseudo_target_station_id"].astype(str)
    squared = frame["transfer_nrmse"] ** 2

    source_totals = frame.groupby(source)["transfer_nrmse"].sum()
    source_sq_totals = squared.groupby(source).sum()
    source_counts = frame.groupby(source)["transfer_nrmse"].count()
    pair_totals = frame.groupby([source, pseudo_target])["transfer_nrmse"].sum()
    pair_sq_totals = squared.groupby([source, pseudo_target]).sum()
    pair_counts = frame.groupby([source, pseudo_target])["transfer_nrmse"].count()

    total_for_source = source.map(source_totals).to_numpy(float)
    sq_total_for_source = source.map(source_sq_totals).to_numpy(float)
    count_for_source = source.map(source_counts).to_numpy(float)
    pair_index = pd.MultiIndex.from_arrays([source, pseudo_target])
    own_pair_total = pair_totals.reindex(pair_index).to_numpy(float)
    own_pair_sq_total = pair_sq_totals.reindex(pair_index).to_numpy(float)
    own_pair_count = pair_counts.reindex(pair_index).to_numpy(float)

    remaining_total = total_for_source - own_pair_total
    remaining_sq_total = sq_total_for_source - own_pair_sq_total
    remaining_count = count_for_source - own_pair_count
    if (remaining_count <= 0).any():
        raise ValueError(
            "source prior std requires at least one other pseudo-target "
            "per source"
        )
    remaining_mean = remaining_total / remaining_count
    remaining_variance = remaining_sq_total / remaining_count - remaining_mean**2
    remaining_variance = np.clip(remaining_variance, 0.0, None)
    result["source_prior_error_std"] = np.sqrt(remaining_variance)
    return result


def validate_meta_training_frame(
    frame: pd.DataFrame,
    *,
    active_target: str,
    test_event: str,
) -> None:
    missing = set(META_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing meta columns: {sorted(missing)}")
    key_columns = [
        "source_station_id",
        "pseudo_target_station_id",
        "event_id",
        "seed",
    ]
    if frame[key_columns].duplicated().any():
        raise ValueError("duplicate meta-training rows")
    if active_target in set(frame["source_station_id"]) or active_target in set(
        frame["pseudo_target_station_id"]
    ):
        raise ValueError("active target appears in meta-training labels")
    if test_event in set(frame["event_id"]):
        raise ValueError("test event appears in meta-training labels")
    numeric = frame[[*FEATURE_COLUMNS, "transfer_nrmse"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("meta-training distances and labels must be finite")
    if (frame[list(FEATURE_COLUMNS)] < 0).any().any():
        raise ValueError("distance features must be nonnegative")
    if (frame["transfer_nrmse"] < 0).any():
        raise ValueError("transfer_nrmse must be nonnegative")
