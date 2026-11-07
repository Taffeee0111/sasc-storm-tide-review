"""Reusable source residual training and target-blind prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from src.experiments.minimal_dataset import (
    SourceScalers,
    build_event_windows,
    fit_source_scalers,
    inverse_transform_labels,
    transform_frame,
)
from src.experiments.minimal_runner import _predict, _train_model
from src.models.bilstm_attention import BiLSTMAttention


TARGET_LABELS = {"water_level_m", "ntr_m"}


def build_target_inputs(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Return only deployable target inputs in a fixed order."""
    required = {"event_id", "timestamp_utc", *features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing target input columns: {missing}")
    result = frame[["event_id", "timestamp_utc", *features]].copy()
    result["timestamp_utc"] = pd.to_datetime(
        result["timestamp_utc"], utc=True, errors="raise"
    )
    if result["timestamp_utc"].duplicated().any():
        raise ValueError("target input timestamps must be unique")
    values = result[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("target inputs must be complete and finite")
    return result


def reconstruct_water_level(
    timestamps,
    *,
    predicted_residual,
    target_inputs: pd.DataFrame,
) -> np.ndarray:
    """Add target astronomical tide to a residual prediction."""
    times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="raise"))
    target = target_inputs[["timestamp_utc", "predicted_tide_m"]].copy()
    target["timestamp_utc"] = pd.to_datetime(
        target["timestamp_utc"], utc=True, errors="raise"
    )
    tide = target.set_index("timestamp_utc")["predicted_tide_m"].reindex(times)
    if tide.isna().any():
        raise ValueError("target predicted tide does not align with predictions")
    residual = np.asarray(predicted_residual, dtype=float)
    if len(residual) != len(times):
        raise ValueError("residual prediction length does not match timestamps")
    return residual + tide.to_numpy(float)


def save_prediction_checkpoint(
    path: Path,
    *,
    state_dict: dict,
    metadata: dict[str, Any],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    label_mean: np.ndarray,
    label_scale: np.ndarray,
    model_config: dict,
) -> None:
    """Save the auditable residual checkpoint contract."""
    required = {
        "source_station_id",
        "test_event",
        "train_events",
        "validation_event",
        "seed",
        "features",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"missing checkpoint metadata: {missing}")
    payload = {
        "state_dict": state_dict,
        **metadata,
        "source_training_label": "ntr_m",
        "reconstruct_water_level_with_predicted_tide": True,
        "feature_scaler_mean": np.asarray(feature_mean, dtype=float),
        "feature_scaler_scale": np.asarray(feature_scale, dtype=float),
        "label_scaler_mean": np.asarray(label_mean, dtype=float),
        "label_scaler_scale": np.asarray(label_scale, dtype=float),
        "model_config": dict(model_config),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_residual_checkpoint(
    data: pd.DataFrame,
    *,
    source_station_id: str,
    train_events: list[str],
    validation_event: str,
    test_event: str,
    features: list[str],
    model_config: dict,
    seed: int,
    checkpoint_path: Path,
    history_path: Path,
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train one event-held-out source residual model and save it."""
    source = data.loc[data["station_id"].astype(str) == str(source_station_id)]
    training = source.loc[source["event_id"].isin(train_events)].copy()
    validation = source.loc[source["event_id"] == validation_event].copy()
    if training.empty or validation.empty:
        raise ValueError("source training and validation rows are required")
    scalers = fit_source_scalers(training, features, label_column="ntr_m")
    training_scaled = transform_frame(
        training, features, scalers, transform_label=True
    )
    validation_scaled = transform_frame(
        validation, features, scalers, transform_label=True
    )
    lookback = int(model_config["lookback_hours"])
    x_train, y_train, _ = build_event_windows(
        training_scaled,
        features,
        label_column="ntr_m",
        lookback=lookback,
        pad_initial=True,
    )
    x_validation, y_validation, _ = build_event_windows(
        validation_scaled,
        features,
        label_column="ntr_m",
        lookback=lookback,
        pad_initial=True,
    )
    if y_train is None or y_validation is None:
        raise RuntimeError("source residual labels are unavailable")
    model, history = _train_model(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        model_config=model_config,
        input_size=len(features),
        seed=int(seed),
        max_epochs=(
            int(max_epochs_override)
            if max_epochs_override is not None
            else int(model_config["max_epochs"])
        ),
    )
    save_prediction_checkpoint(
        checkpoint_path,
        state_dict=model.state_dict(),
        metadata={
            "source_station_id": str(source_station_id),
            "test_event": str(test_event),
            "train_events": list(train_events),
            "validation_event": str(validation_event),
            "seed": int(seed),
            "features": list(features),
        },
        feature_mean=scalers.feature_scaler.mean_,
        feature_scale=scalers.feature_scaler.scale_,
        label_mean=scalers.label_scaler.mean_,
        label_scale=scalers.label_scaler.scale_,
        model_config=model_config,
    )
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "epochs_ran": len(history),
        "best_validation_mse_scaled": min(
            row["validation_mse"] for row in history
        ),
    }


def _standard_scaler(mean: np.ndarray, scale: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(mean, dtype=float)
    scaler.scale_ = np.asarray(scale, dtype=float)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = len(scaler.mean_)
    scaler.n_samples_seen_ = 1
    return scaler


def predict_residual_checkpoint(
    checkpoint_path: Path,
    target_inputs: pd.DataFrame,
) -> pd.DataFrame:
    """Predict target residual and reconstructed water level without labels."""
    if TARGET_LABELS & set(target_inputs.columns):
        raise ValueError("target labels are forbidden in blind prediction inputs")
    checkpoint = torch.load(Path(checkpoint_path), weights_only=False)
    features = list(checkpoint["features"])
    inputs = build_target_inputs(target_inputs, features)
    model_config = checkpoint["model_config"]
    model = BiLSTMAttention(
        input_size=len(features),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    scalers = SourceScalers(
        feature_columns=tuple(features),
        label_column="ntr_m",
        feature_scaler=_standard_scaler(
            checkpoint["feature_scaler_mean"],
            checkpoint["feature_scaler_scale"],
        ),
        label_scaler=_standard_scaler(
            checkpoint["label_scaler_mean"],
            checkpoint["label_scaler_scale"],
        ),
    )
    scaled = transform_frame(inputs, features, scalers, transform_label=False)
    x, labels, times = build_event_windows(
        scaled,
        features,
        label_column=None,
        lookback=int(model_config["lookback_hours"]),
        pad_initial=True,
    )
    if labels is not None:
        raise RuntimeError("target labels crossed the prediction boundary")
    residual_scaled = _predict(
        model,
        x,
        batch_size=int(model_config["batch_size"]),
    )
    residual = inverse_transform_labels(residual_scaled, scalers)
    water_level = reconstruct_water_level(
        times,
        predicted_residual=residual,
        target_inputs=inputs,
    )
    return pd.DataFrame(
        {
            "timestamp_utc": times,
            "predicted_ntr_m": residual,
            "predicted_water_level_m": water_level,
        }
    )
