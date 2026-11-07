"""Evaluation boundary that reveals target labels only after prediction save."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.minimal_methods import METHOD_COLUMNS
from src.metrics import evaluate_event


def evaluate_saved_predictions(
    prediction_path: Path,
    label_frame: pd.DataFrame,
    *,
    target_station_id: str,
    event_id: str,
    seed: int,
) -> pd.DataFrame:
    """Evaluate a previously saved method forecast against target labels."""
    prediction_path = Path(prediction_path)
    if not prediction_path.is_file():
        raise ValueError("prediction file must exist before target labels are revealed")
    predictions = pd.read_csv(prediction_path)
    required_predictions = {"timestamp_utc", *METHOD_COLUMNS}
    missing_predictions = sorted(required_predictions - set(predictions.columns))
    if missing_predictions:
        raise ValueError(f"missing prediction columns: {missing_predictions}")
    required_labels = {"timestamp_utc", "water_level_m"}
    missing_labels = sorted(required_labels - set(label_frame.columns))
    if missing_labels:
        raise ValueError(f"missing target label columns: {missing_labels}")

    predictions = predictions.copy()
    labels = label_frame[["timestamp_utc", "water_level_m"]].copy()
    predictions["timestamp_utc"] = pd.to_datetime(
        predictions["timestamp_utc"], utc=True, errors="raise"
    )
    labels["timestamp_utc"] = pd.to_datetime(
        labels["timestamp_utc"], utc=True, errors="raise"
    )
    if predictions["timestamp_utc"].duplicated().any():
        raise ValueError("prediction timestamps must be unique")
    if labels["timestamp_utc"].duplicated().any():
        raise ValueError("label timestamps must be unique")

    prediction_times = pd.Index(predictions["timestamp_utc"])
    label_times = pd.Index(labels["timestamp_utc"])
    if len(prediction_times) != len(label_times) or set(prediction_times) != set(
        label_times
    ):
        raise ValueError("prediction and target label timestamps must match exactly")
    merged = predictions.merge(
        labels,
        on="timestamp_utc",
        how="inner",
        validate="one_to_one",
    ).sort_values("timestamp_utc")

    rows = []
    for method in METHOD_COLUMNS:
        metrics = evaluate_event(
            merged["water_level_m"].to_numpy(float),
            merged[method].to_numpy(float),
            merged["timestamp_utc"],
        )
        rows.append(
            {
                "target_station_id": str(target_station_id),
                "event_id": str(event_id),
                "seed": int(seed),
                "method": method,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def evaluate_saved_source_predictions(
    prediction_path: Path,
    label_frame: pd.DataFrame,
    *,
    source_station_ids: list[str],
    target_station_id: str,
    event_id: str,
    seed: int,
) -> pd.DataFrame:
    """Evaluate each previously saved raw source forecast."""
    prediction_path = Path(prediction_path)
    if not prediction_path.is_file():
        raise ValueError("source prediction file must exist before evaluation")
    predictions = pd.read_csv(prediction_path)
    source_station_ids = [str(value) for value in source_station_ids]
    required = {"timestamp_utc", *source_station_ids}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"missing source prediction columns: {missing}")
    if not {"timestamp_utc", "water_level_m"}.issubset(label_frame.columns):
        raise ValueError("target labels require timestamp_utc and water_level_m")
    predictions["timestamp_utc"] = pd.to_datetime(
        predictions["timestamp_utc"], utc=True, errors="raise"
    )
    labels = label_frame[["timestamp_utc", "water_level_m"]].copy()
    labels["timestamp_utc"] = pd.to_datetime(
        labels["timestamp_utc"], utc=True, errors="raise"
    )
    if predictions["timestamp_utc"].duplicated().any():
        raise ValueError("source prediction timestamps must be unique")
    if labels["timestamp_utc"].duplicated().any():
        raise ValueError("label timestamps must be unique")
    if len(predictions) != len(labels) or set(predictions["timestamp_utc"]) != set(
        labels["timestamp_utc"]
    ):
        raise ValueError("source prediction and target label timestamps must match")
    merged = predictions.merge(
        labels,
        on="timestamp_utc",
        how="inner",
        validate="one_to_one",
    ).sort_values("timestamp_utc")
    rows = []
    for source_station_id in source_station_ids:
        rows.append(
            {
                "target_station_id": str(target_station_id),
                "event_id": str(event_id),
                "seed": int(seed),
                "source_station_id": source_station_id,
                **evaluate_event(
                    merged["water_level_m"].to_numpy(float),
                    merged[source_station_id].to_numpy(float),
                    merged["timestamp_utc"],
                ),
            }
        )
    return pd.DataFrame(rows)
