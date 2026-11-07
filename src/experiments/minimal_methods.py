"""Fair source-selection and source-fusion methods."""

from __future__ import annotations

import numpy as np
import pandas as pd


METHOD_COLUMNS = [
    "nearest_source",
    "equal_ensemble",
    "similarity_top1",
    "similarity_weighted",
]
FORBIDDEN_LABEL_COLUMNS = {
    "water_level_m",
    "ntr_m",
    "observed_water_level_m",
    "target_error",
}


def combine_predictions(
    source_predictions: pd.DataFrame,
    source_information: pd.DataFrame,
) -> pd.DataFrame:
    """Apply four methods to the same three raw source forecasts."""
    forbidden = sorted(FORBIDDEN_LABEL_COLUMNS & set(source_predictions.columns))
    if forbidden:
        raise ValueError(f"target label columns are forbidden: {forbidden}")
    required_info = {
        "source_station_id",
        "geo_distance_km",
        "combined_distance",
        "similarity_weight",
    }
    missing_info = sorted(required_info - set(source_information.columns))
    if missing_info:
        raise ValueError(f"missing source information columns: {missing_info}")

    information = source_information.copy()
    information["source_station_id"] = information["source_station_id"].astype(str)
    if len(information) != 3 or information["source_station_id"].nunique() != 3:
        raise ValueError("exactly three unique source records are required")
    source_ids = information["source_station_id"].tolist()
    missing_predictions = sorted(set(source_ids) - set(source_predictions.columns))
    if missing_predictions:
        raise ValueError(f"missing source prediction columns: {missing_predictions}")
    if "timestamp_utc" not in source_predictions:
        raise ValueError("source predictions require timestamp_utc")

    predictions = source_predictions[source_ids].astype(float)
    if not np.isfinite(predictions.to_numpy()).all():
        raise ValueError("source predictions must be finite")
    weights = information.set_index("source_station_id")[
        "similarity_weight"
    ].reindex(source_ids).to_numpy(float)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("similarity weights must be finite and non-negative")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("similarity weights must sum to one")

    nearest_source = str(
        information.loc[information["geo_distance_km"].astype(float).idxmin()][
            "source_station_id"
        ]
    )
    similarity_top1 = str(
        information.loc[information["combined_distance"].astype(float).idxmin()][
            "source_station_id"
        ]
    )

    result = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                source_predictions["timestamp_utc"], utc=True, errors="raise"
            ),
            "nearest_source": predictions[nearest_source].to_numpy(float),
            "equal_ensemble": predictions.mean(axis=1).to_numpy(float),
            "similarity_top1": predictions[similarity_top1].to_numpy(float),
            "similarity_weighted": predictions.to_numpy(float) @ weights,
        }
    )
    if result["timestamp_utc"].duplicated().any():
        raise ValueError("prediction timestamps must be unique")
    return result
