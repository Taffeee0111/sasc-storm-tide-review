"""Source-only calibration of atmosphere/wave/coastal similarity weights."""

from __future__ import annotations

from itertools import product
import math

import numpy as np
import pandas as pd

from src.preexperiment.similarity import haversine_km


BEAUFORT_ID = "8656483"
GROUPS = ("atmosphere", "wave", "coastal")
SUMMARY_STATISTICS = ("mean", "std", "q05", "q50", "q95")


def simplex_grid(step: float) -> list[tuple[float, float, float]]:
    """Return a deterministic three-component simplex grid."""
    reciprocal = round(1.0 / step)
    if step <= 0 or not np.isclose(reciprocal * step, 1.0):
        raise ValueError("simplex step must evenly divide one")
    return [
        (i / reciprocal, j / reciprocal, k / reciprocal)
        for i in range(reciprocal + 1)
        for j in range(reciprocal + 1 - i)
        for k in [reciprocal - i - j]
    ]


def calculate_calibrated_weights(
    source_information: pd.DataFrame,
    *,
    group_weights: tuple[float, float, float],
    temperature: float,
) -> pd.Series:
    """Apply frozen group weights without geographic or target-label inputs."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if "target_station_id" in source_information and (
        source_information["target_station_id"].astype(str) == BEAUFORT_ID
    ).any():
        raise ValueError("Beaufort is forbidden in source-only calibration")
    if any(value < 0 for value in group_weights) or not np.isclose(
        sum(group_weights), 1.0
    ):
        raise ValueError("group weights must be nonnegative and sum to one")
    columns = [f"{group}_distance" for group in GROUPS]
    missing = sorted({"source_station_id", *columns} - set(source_information))
    if missing:
        raise ValueError(f"missing similarity columns: {missing}")
    values = source_information[columns].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("similarity distances must be finite")
    combined = values @ np.asarray(group_weights, dtype=float)
    logits = -combined / float(temperature)
    logits -= logits.max()
    scores = np.exp(logits)
    return pd.Series(
        scores / scores.sum(),
        index=source_information["source_station_id"].astype(str),
        name="similarity_weight",
    )


def select_calibration(candidate_scores: pd.DataFrame) -> dict[str, float]:
    """Select minimum NRMSE with deterministic non-extreme tie-breaking."""
    required = {
        "w_atmosphere",
        "w_wave",
        "w_coastal",
        "temperature",
        "mean_nrmse",
    }
    missing = sorted(required - set(candidate_scores))
    if missing:
        raise ValueError(f"missing candidate score columns: {missing}")
    frame = candidate_scores.copy()
    if frame.empty or not np.isfinite(frame["mean_nrmse"]).all():
        raise ValueError("candidate scores must be finite and non-empty")

    def entropy(row: pd.Series) -> float:
        values = [
            float(row["w_atmosphere"]),
            float(row["w_wave"]),
            float(row["w_coastal"]),
        ]
        return -sum(value * math.log(value) for value in values if value > 0)

    frame["_entropy"] = frame.apply(entropy, axis=1)
    best_score = float(frame["mean_nrmse"].min())
    tied = frame[np.isclose(frame["mean_nrmse"], best_score)].copy()
    selected = tied.sort_values(
        ["_entropy", "temperature"],
        ascending=[False, False],
    ).iloc[0]
    return {
        "w_atmosphere": float(selected["w_atmosphere"]),
        "w_wave": float(selected["w_wave"]),
        "w_coastal": float(selected["w_coastal"]),
        "temperature": float(selected["temperature"]),
        "mean_nrmse": float(selected["mean_nrmse"]),
    }


def _descriptor(frame: pd.DataFrame, variables: list[str]) -> pd.Series:
    result: dict[str, float] = {}
    for variable in variables:
        values = pd.to_numeric(frame[variable], errors="coerce").to_numpy(float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"incomplete descriptor variable: {variable}")
        result[f"{variable}__mean"] = float(np.mean(values))
        result[f"{variable}__std"] = float(np.std(values))
        result[f"{variable}__q05"] = float(np.quantile(values, 0.05))
        result[f"{variable}__q50"] = float(np.quantile(values, 0.50))
        result[f"{variable}__q95"] = float(np.quantile(values, 0.95))
    return pd.Series(result, dtype=float)


def fit_descriptor_reference(
    events: pd.DataFrame,
    coastal: pd.DataFrame,
    *,
    source_station_ids: list[str],
    calibration_events: list[str],
    group_variables: dict[str, list[str]],
    coastal_reference_station_ids: list[str] | None = None,
) -> dict:
    """Fit feature scaling exclusively on source/calibration descriptors.

    ``coastal_reference_station_ids`` lets the coastal group's mean/scale be
    fit from a station set other than ``source_station_ids`` (e.g. a wider
    built-station inventory), without touching the event-dependent
    atmosphere/wave references. It defaults to ``source_station_ids``, which
    reproduces the original behavior exactly.
    """
    events = events.copy()
    events["station_id"] = events["station_id"].astype(str)
    coastal = coastal.copy()
    coastal["station_id"] = coastal["station_id"].astype(str)
    reference: dict[str, dict] = {}
    for group in ("atmosphere", "wave"):
        rows = []
        for station_id, event_id in product(
            source_station_ids, calibration_events
        ):
            frame = events.loc[
                (events["station_id"] == station_id)
                & (events["event_id"] == event_id)
            ]
            if len(frame) != 216:
                raise ValueError(
                    f"{station_id}/{event_id} requires 216 descriptor rows"
                )
            rows.append(_descriptor(frame, list(group_variables[group])))
        table = pd.DataFrame(rows)
        scale = table.std(axis=0, ddof=0).replace(0.0, 1.0)
        reference[group] = {
            "columns": list(table.columns),
            "mean": table.mean(axis=0).tolist(),
            "scale": scale.tolist(),
        }
    coastal_columns = list(group_variables["coastal"])
    coastal_station_ids = (
        source_station_ids
        if coastal_reference_station_ids is None
        else coastal_reference_station_ids
    )
    table = coastal.set_index("station_id").loc[
        coastal_station_ids, coastal_columns
    ].astype(float)
    reference["coastal"] = {
        "columns": coastal_columns,
        "mean": table.mean(axis=0).tolist(),
        "scale": table.std(axis=0, ddof=0).replace(0.0, 1.0).tolist(),
    }
    return reference


def build_pair_distances(
    events: pd.DataFrame,
    coastal: pd.DataFrame,
    *,
    candidate_source_ids: list[str],
    target_station_id: str,
    event_id: str,
    group_variables: dict[str, list[str]],
    reference: dict,
) -> pd.DataFrame:
    """Build target-label-free distances using a frozen source-only reference."""
    events = events.copy()
    events["station_id"] = events["station_id"].astype(str)
    coastal = coastal.copy()
    coastal["station_id"] = coastal["station_id"].astype(str)
    station_ids = candidate_source_ids + [str(target_station_id)]
    descriptors: dict[str, dict[str, pd.Series]] = {}
    for group in ("atmosphere", "wave"):
        descriptors[group] = {}
        for station_id in station_ids:
            frame = events.loc[
                (events["station_id"] == station_id)
                & (events["event_id"] == event_id)
            ]
            if len(frame) != 216:
                raise ValueError(f"{station_id}/{event_id} requires 216 rows")
            descriptors[group][station_id] = _descriptor(
                frame, list(group_variables[group])
            )
    coastal_table = coastal.set_index("station_id")
    first = events.loc[
        (events["station_id"] == str(target_station_id))
        & (events["event_id"] == event_id)
    ].iloc[0]
    rows = []
    for source_id in candidate_source_ids:
        source_first = events.loc[
            (events["station_id"] == source_id)
            & (events["event_id"] == event_id)
        ].iloc[0]
        row = {
            "source_station_id": source_id,
            "target_station_id": str(target_station_id),
            "event_id": event_id,
            "geo_distance_km": haversine_km(
                float(source_first["latitude"]),
                float(source_first["longitude"]),
                float(first["latitude"]),
                float(first["longitude"]),
            ),
        }
        for group in GROUPS:
            item = reference[group]
            columns = list(item["columns"])
            mean = np.asarray(item["mean"], dtype=float)
            scale = np.asarray(item["scale"], dtype=float)
            if group == "coastal":
                source_values = coastal_table.loc[source_id, columns].to_numpy(
                    float
                )
                target_values = coastal_table.loc[
                    str(target_station_id), columns
                ].to_numpy(float)
            else:
                source_values = descriptors[group][source_id].reindex(
                    columns
                ).to_numpy(float)
                target_values = descriptors[group][
                    str(target_station_id)
                ].reindex(columns).to_numpy(float)
            difference = (source_values - mean) / scale - (
                target_values - mean
            ) / scale
            row[f"{group}_distance"] = float(
                np.sqrt(np.mean(difference**2))
            )
        rows.append(row)
    return pd.DataFrame(rows)
