"""Leakage-safe scaling and event-window construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FORBIDDEN_TARGET_COLUMNS = {
    "water_level_m",
    "ntr_m",
    "observed_water_level_m",
    "target_error",
    "rmse",
    "nse",
    "kge",
}


@dataclass
class SourceScalers:
    """Scalers fitted exclusively on one source station's training rows."""

    feature_columns: tuple[str, ...]
    label_column: str
    feature_scaler: StandardScaler
    label_scaler: StandardScaler


def _validate_features(frame: pd.DataFrame, feature_columns: list[str]) -> None:
    forbidden = sorted(set(feature_columns) & FORBIDDEN_TARGET_COLUMNS)
    if forbidden:
        raise ValueError(f"forbidden target feature columns: {forbidden}")
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    values = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("model features must be complete and finite")


def fit_source_scalers(
    source_training_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    label_column: str,
) -> SourceScalers:
    """Fit feature and label scalers on source training rows only."""
    if source_training_frame.empty:
        raise ValueError("source training frame must not be empty")
    _validate_features(source_training_frame, feature_columns)
    if label_column not in source_training_frame:
        raise ValueError(f"missing source label column: {label_column}")
    label = pd.to_numeric(
        source_training_frame[label_column], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(label).all():
        raise ValueError("source training labels must be complete and finite")
    feature_scaler = StandardScaler().fit(
        source_training_frame[feature_columns].to_numpy(float)
    )
    label_scaler = StandardScaler().fit(label.reshape(-1, 1))
    return SourceScalers(
        feature_columns=tuple(feature_columns),
        label_column=label_column,
        feature_scaler=feature_scaler,
        label_scaler=label_scaler,
    )


def transform_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    scalers: SourceScalers,
    *,
    transform_label: bool = False,
) -> pd.DataFrame:
    """Transform features with source-fitted parameters."""
    if tuple(feature_columns) != scalers.feature_columns:
        raise ValueError("feature columns do not match fitted source scaler")
    _validate_features(frame, feature_columns)
    result = frame.copy()
    result.loc[:, feature_columns] = scalers.feature_scaler.transform(
        result[feature_columns].to_numpy(float)
    )
    if transform_label:
        if scalers.label_column not in result:
            raise ValueError("label is required when transform_label is true")
        label = pd.to_numeric(result[scalers.label_column], errors="coerce").to_numpy(
            float
        )
        if not np.isfinite(label).all():
            raise ValueError("labels must be complete and finite")
        result.loc[:, scalers.label_column] = scalers.label_scaler.transform(
            label.reshape(-1, 1)
        ).ravel()
    return result


def inverse_transform_labels(values, scalers: SourceScalers) -> np.ndarray:
    """Return physical-metre predictions from standardized outputs."""
    array = np.asarray(values, dtype=float).reshape(-1, 1)
    return scalers.label_scaler.inverse_transform(array).ravel()


def build_event_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    label_column: str | None,
    lookback: int,
    pad_initial: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, pd.DatetimeIndex]:
    """Build one-step-ahead windows without crossing event boundaries."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    required = {"event_id", "timestamp_utc", *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing window columns: {missing}")
    _validate_features(frame, feature_columns)
    if label_column is not None and label_column not in frame:
        raise ValueError(f"missing label column: {label_column}")

    windows: list[np.ndarray] = []
    labels: list[float] = []
    times: list[pd.Timestamp] = []
    for _, event_rows in frame.groupby("event_id", sort=False):
        event_rows = event_rows.copy()
        event_rows["timestamp_utc"] = pd.to_datetime(
            event_rows["timestamp_utc"], utc=True, errors="raise"
        )
        event_rows = event_rows.sort_values("timestamp_utc")
        if event_rows["timestamp_utc"].duplicated().any():
            raise ValueError("event windows cannot contain duplicate timestamps")
        deltas = event_rows["timestamp_utc"].diff().dropna()
        if not deltas.empty and not (deltas == pd.Timedelta(hours=1)).all():
            raise ValueError("event windows require continuous hourly timestamps")
        features = event_rows[feature_columns].to_numpy(dtype=np.float32)
        label_values = (
            None
            if label_column is None
            else pd.to_numeric(event_rows[label_column], errors="coerce").to_numpy(
                float
            )
        )
        if label_values is not None and not np.isfinite(label_values).all():
            raise ValueError("window labels must be complete and finite")
        timestamps = pd.DatetimeIndex(event_rows["timestamp_utc"])
        if pad_initial:
            for index in range(len(event_rows)):
                start = max(0, index - lookback + 1)
                window = features[start : index + 1]
                if len(window) < lookback:
                    padding = np.repeat(
                        window[[0]], repeats=lookback - len(window), axis=0
                    )
                    window = np.concatenate([padding, window], axis=0)
                windows.append(window)
                if label_values is not None:
                    labels.append(float(label_values[index]))
                times.append(timestamps[index])
        else:
            for stop in range(lookback, len(event_rows)):
                windows.append(features[stop - lookback : stop])
                if label_values is not None:
                    labels.append(float(label_values[stop]))
                times.append(timestamps[stop])

    if windows:
        x = np.stack(windows).astype(np.float32)
    else:
        x = np.empty((0, lookback, len(feature_columns)), dtype=np.float32)
    y = None if label_column is None else np.asarray(labels, dtype=np.float32)
    return x, y, pd.DatetimeIndex(times)
