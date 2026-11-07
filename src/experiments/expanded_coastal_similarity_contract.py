"""Meta-training contract for the expanded-coastal learned similarity
metric. Extends seven_station_learned_similarity_contract with two new,
target-blind, purely static features: water_fraction_50km and
shelf_mean_depth_m -- both already present in coastal_static_features.csv
but never included in coastal_distance's five-descriptor RMS combination.
Retrospective correlation checks against results/seven_station_learned_similarity
meta-training data showed both carry real, largely non-redundant signal
with transfer_nrmse (water_fraction r=0.196, shelf_mean_depth r=0.234,
vs coastal_distance's own r~0.20-0.24; redundancy with coastal_distance
r=-0.088), unlike storm_shape_distance (Round 4, r=-0.018) which never
showed signal. Both descriptors are event-invariant (zero variance within
a fixed source/target station pair across events), consistent with the
diagnosed pattern that only static, not event-varying, features have
worked so far."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.seven_station_learned_similarity_contract import (
    ALPHA_GRID,
    LAMBDA_GRID,
    TEMPERATURE_GRID,
    attach_source_quality_prior,
    build_allowed_meta_grid,
)

STATIC_DESCRIPTOR_COLUMNS = ("water_fraction_50km", "shelf_mean_depth_m")

FEATURE_COLUMNS = (
    "atmosphere_distance",
    "wave_distance",
    "coastal_distance",
    "source_prior_error",
    "water_fraction_distance",
    "shelf_depth_distance",
)
META_COLUMNS = (
    "source_station_id",
    "pseudo_target_station_id",
    "event_id",
    "seed",
    *FEATURE_COLUMNS,
    "transfer_nrmse",
)

__all__ = [
    "STATIC_DESCRIPTOR_COLUMNS",
    "FEATURE_COLUMNS",
    "ALPHA_GRID",
    "TEMPERATURE_GRID",
    "LAMBDA_GRID",
    "META_COLUMNS",
    "build_allowed_meta_grid",
    "attach_source_quality_prior",
    "compute_static_descriptor_distance",
    "validate_meta_training_frame",
]


def compute_static_descriptor_distance(
    coastal_full: pd.DataFrame,
    *,
    source_station_id: str,
    target_station_id: str,
    column: str,
    reference_station_ids: list[str],
) -> float:
    """Standardized absolute distance for one static descriptor column.

    ``reference_station_ids`` fixes the standardization mean/scale to the
    full built-station inventory excluding the active target (matching the
    coastal-group widened-reference convention used since Round 2), never
    the active target's own value. Purely a function of ``coastal_full``;
    reads no event data, no water_level_m/ntr_m.
    """
    table = coastal_full.set_index(
        coastal_full["station_id"].astype(str)
    )[column].astype(float)
    reference = table.loc[[str(s) for s in reference_station_ids]]
    mean = float(reference.mean())
    scale = float(reference.std(ddof=0))
    scale = scale if scale > 1e-9 else 1.0
    source_value = float(table.loc[str(source_station_id)])
    target_value = float(table.loc[str(target_station_id)])
    return abs((source_value - mean) / scale - (target_value - mean) / scale)


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
