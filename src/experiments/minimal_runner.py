"""End-to-end runner for the frozen three-source one-target pilot."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.minimal_contract import build_splits
from src.experiments.minimal_dataset import (
    build_event_windows,
    fit_source_scalers,
    inverse_transform_labels,
    transform_frame,
)
from src.experiments.minimal_evaluation import (
    evaluate_saved_predictions,
    evaluate_saved_source_predictions,
)
from src.experiments.minimal_methods import combine_predictions
from src.models.bilstm_attention import BiLSTMAttention


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _validation_loss(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> float:
    model.eval()
    losses: list[float] = []
    criterion = nn.MSELoss(reduction="sum")
    loader = _loader(x, y, batch_size=batch_size, shuffle=False, seed=0)
    with torch.no_grad():
        for inputs, labels in loader:
            losses.append(float(criterion(model(inputs), labels).item()))
    return float(sum(losses) / len(y))


def _train_model(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    model_config: dict,
    input_size: int,
    seed: int,
    max_epochs: int,
) -> tuple[BiLSTMAttention, list[dict[str, float]]]:
    _set_seed(seed)
    model = BiLSTMAttention(
        input_size=input_size,
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(model_config["learning_rate"])
    )
    criterion = nn.MSELoss()
    batch_size = int(model_config["batch_size"])
    patience = int(model_config["early_stopping_patience"])
    loader = _loader(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for inputs, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_count += len(labels)
        training_loss = total_loss / total_count
        validation_loss = _validation_loss(
            model,
            x_validation,
            y_validation,
            batch_size,
        )
        history.append(
            {
                "epoch": float(epoch),
                "training_mse": float(training_loss),
                "validation_mse": float(validation_loss),
            }
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    return model, history


def _predict(
    model: nn.Module,
    x: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    with torch.no_grad():
        for (inputs,) in loader:
            outputs.append(model(inputs).cpu().numpy())
    return np.concatenate(outputs).astype(float)


def _select_rows(
    data: pd.DataFrame,
    *,
    station_id: str,
    events: list[str],
    expected_hours_per_event: int,
) -> pd.DataFrame:
    rows = data[
        (data["station_id"] == station_id) & data["event_id"].isin(events)
    ].copy()
    expected = len(events) * expected_hours_per_event
    if len(rows) != expected:
        raise ValueError(
            f"{station_id}/{events} expected {expected} rows, found {len(rows)}"
        )
    return rows


def _write_event_figure(
    output_path: Path,
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    event_id: str,
) -> None:
    merged = predictions.merge(
        labels[["timestamp_utc", "water_level_m"]],
        on="timestamp_utc",
        how="inner",
        validate="one_to_one",
    )
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        merged["timestamp_utc"],
        merged["water_level_m"],
        color="black",
        linewidth=2.0,
        label="Observed",
    )
    for column, style in [
        ("nearest_source", "--"),
        ("equal_ensemble", ":"),
        ("similarity_top1", "-."),
        ("similarity_weighted", "-"),
    ]:
        axis.plot(
            merged["timestamp_utc"],
            merged[column],
            linestyle=style,
            linewidth=1.2,
            label=column,
        )
    axis.set_title(f"Kiptopeke — {event_id}")
    axis.set_ylabel("Water level (m)")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    figure.autofmt_xdate()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_experiment(
    *,
    config_path: Path,
    data_path: Path,
    similarity_path: Path,
    output_dir: Path,
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train source models, save forecasts, then evaluate target labels."""
    started = datetime.now(timezone.utc)
    config_path = Path(config_path)
    data_path = Path(data_path)
    similarity_path = Path(similarity_path)
    output_dir = Path(output_dir)
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    if config.get("data_policy", {}).get("allow_new_downloads") is not False:
        raise ValueError("minimal experiment must forbid new downloads")
    if (
        config.get("data_policy", {}).get("target_labels_evaluation_only")
        is not True
    ):
        raise ValueError("target labels must be evaluation-only")
    splits = build_splits(config)
    features = list(config["features"]["dynamic"])
    forbidden = set(config["features"]["forbidden_target"])
    if forbidden & set(features):
        raise ValueError("target labels are present in model features")

    data = pd.read_csv(
        data_path,
        dtype={"station_id": str, "event_id": str},
        parse_dates=["timestamp_utc"],
    )
    similarity = pd.read_csv(
        similarity_path,
        dtype={
            "source_station_id": str,
            "target_station_id": str,
            "test_event": str,
        },
    )
    checkpoint_dir = output_dir / "checkpoints"
    prediction_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    history_dir = output_dir / "training_history"
    figure_dir = output_dir / "figures"
    for directory in [
        checkpoint_dir,
        prediction_dir,
        metrics_dir,
        history_dir,
        figure_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    metrics_frames = []
    source_metrics_frames = []
    checkpoint_records: list[dict[str, Any]] = []
    prediction_hours: list[int] = []
    model_config = config["model"]
    source_label = str(model_config.get("label_column", "water_level_m"))
    reconstruct_water_level = bool(
        model_config.get(
            "reconstruct_water_level_with_predicted_tide",
            False,
        )
    )
    if source_label not in {"water_level_m", "ntr_m"}:
        raise ValueError("source training label must be water_level_m or ntr_m")
    if reconstruct_water_level != (source_label == "ntr_m"):
        raise ValueError(
            "ntr_m training requires predicted-tide reconstruction and "
            "water_level_m training forbids it"
        )
    max_epochs = (
        int(max_epochs_override)
        if max_epochs_override is not None
        else int(model_config["max_epochs"])
    )
    target_id = str(config["stations"]["target"])
    expected_hours = int(config["events"]["expected_hours_per_event"])

    for seed in [int(value) for value in model_config["random_seeds"]]:
        for split in splits:
            event_id = str(split["test_event"])
            target_rows = _select_rows(
                data,
                station_id=target_id,
                events=[event_id],
                expected_hours_per_event=expected_hours,
            ).sort_values("timestamp_utc")
            target_input = target_rows[
                ["event_id", "timestamp_utc", *features]
            ].copy()
            source_prediction_frame: pd.DataFrame | None = None

            for source_id in split["source_station_ids"]:
                training_rows = _select_rows(
                    data,
                    station_id=source_id,
                    events=list(split["train_events"]),
                    expected_hours_per_event=expected_hours,
                )
                validation_rows = _select_rows(
                    data,
                    station_id=source_id,
                    events=[str(split["validation_event"])],
                    expected_hours_per_event=expected_hours,
                )
                scalers = fit_source_scalers(
                    training_rows,
                    features,
                    label_column=source_label,
                )
                training_scaled = transform_frame(
                    training_rows,
                    features,
                    scalers,
                    transform_label=True,
                )
                validation_scaled = transform_frame(
                    validation_rows,
                    features,
                    scalers,
                    transform_label=True,
                )
                x_train, y_train, _ = build_event_windows(
                    training_scaled,
                    features,
                    label_column=source_label,
                    lookback=int(model_config["lookback_hours"]),
                    pad_initial=True,
                )
                x_validation, y_validation, _ = build_event_windows(
                    validation_scaled,
                    features,
                    label_column=source_label,
                    lookback=int(model_config["lookback_hours"]),
                    pad_initial=True,
                )
                if y_train is None or y_validation is None:
                    raise RuntimeError("source labels unexpectedly unavailable")
                model, history = _train_model(
                    x_train=x_train,
                    y_train=y_train,
                    x_validation=x_validation,
                    y_validation=y_validation,
                    model_config=model_config,
                    input_size=len(features),
                    seed=seed,
                    max_epochs=max_epochs,
                )

                checkpoint_path = (
                    checkpoint_dir
                    / f"{event_id}__source_{source_id}__seed_{seed}.pt"
                )
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "source_station_id": source_id,
                    "target_station_id": target_id,
                    "test_event": event_id,
                    "validation_event": split["validation_event"],
                    "train_events": split["train_events"],
                    "seed": seed,
                    "features": features,
                    "source_training_label": source_label,
                    "reconstruct_water_level_with_predicted_tide": (
                        reconstruct_water_level
                    ),
                    "model_config": dict(model_config),
                    "feature_scaler_mean": scalers.feature_scaler.mean_,
                    "feature_scaler_scale": scalers.feature_scaler.scale_,
                    "label_scaler_mean": scalers.label_scaler.mean_,
                    "label_scaler_scale": scalers.label_scaler.scale_,
                }
                torch.save(checkpoint, checkpoint_path)
                pd.DataFrame(history).to_csv(
                    history_dir
                    / f"{event_id}__source_{source_id}__seed_{seed}.csv",
                    index=False,
                )

                target_scaled = transform_frame(
                    target_input,
                    features,
                    scalers,
                    transform_label=False,
                )
                x_target, target_y, target_times = build_event_windows(
                    target_scaled,
                    features,
                    label_column=None,
                    lookback=int(model_config["lookback_hours"]),
                    pad_initial=True,
                )
                if target_y is not None:
                    raise RuntimeError("target labels crossed the prediction boundary")
                prediction_scaled = _predict(
                    model,
                    x_target,
                    batch_size=int(model_config["batch_size"]),
                )
                prediction = inverse_transform_labels(prediction_scaled, scalers)
                if reconstruct_water_level:
                    target_tide = target_rows.set_index("timestamp_utc")[
                        "predicted_tide_m"
                    ].reindex(target_times)
                    if target_tide.isna().any():
                        raise ValueError(
                            "predicted tide is missing during water-level reconstruction"
                        )
                    prediction = prediction + target_tide.to_numpy(float)
                if source_prediction_frame is None:
                    source_prediction_frame = pd.DataFrame(
                        {"timestamp_utc": target_times}
                    )
                elif not pd.Index(
                    source_prediction_frame["timestamp_utc"]
                ).equals(pd.Index(target_times)):
                    raise ValueError("source prediction timestamps are not aligned")
                source_prediction_frame[source_id] = prediction
                checkpoint_records.append(
                    {
                        "path": checkpoint_path.relative_to(output_dir).as_posix(),
                        "source_station_id": source_id,
                        "test_event": event_id,
                        "seed": seed,
                        "best_validation_mse_scaled": min(
                            row["validation_mse"] for row in history
                        ),
                        "epochs_ran": len(history),
                    }
                )

            if source_prediction_frame is None:
                raise RuntimeError("no source forecasts were generated")
            raw_path = (
                prediction_dir
                / f"kiptopeke_{event_id}_source_predictions.csv"
            )
            source_prediction_frame.to_csv(raw_path, index=False)

            event_similarity = similarity[
                (similarity["target_station_id"] == target_id)
                & (similarity["test_event"] == event_id)
            ].copy()
            methods = combine_predictions(
                source_prediction_frame,
                event_similarity,
            )
            method_path = (
                prediction_dir / f"kiptopeke_{event_id}_methods.csv"
            )
            methods.to_csv(method_path, index=False)
            prediction_hours.append(len(methods))

            evaluation_labels = target_rows[
                ["timestamp_utc", "water_level_m"]
            ].copy()
            metrics_frames.append(
                evaluate_saved_predictions(
                    method_path,
                    evaluation_labels,
                    target_station_id=target_id,
                    event_id=event_id,
                    seed=seed,
                )
            )
            source_metrics_frames.append(
                evaluate_saved_source_predictions(
                    raw_path,
                    evaluation_labels,
                    source_station_ids=list(split["source_station_ids"]),
                    target_station_id=target_id,
                    event_id=event_id,
                    seed=seed,
                )
            )
            _write_event_figure(
                figure_dir / f"kiptopeke_{event_id}.png",
                methods,
                evaluation_labels,
                event_id=event_id,
            )

    metrics = pd.concat(metrics_frames, ignore_index=True)
    metrics_path = metrics_dir / "method_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    source_metrics = pd.concat(source_metrics_frames, ignore_index=True)
    source_metrics.to_csv(metrics_dir / "source_metrics.csv", index=False)
    finished = datetime.now(timezone.utc)
    manifest: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "data_path": str(data_path),
        "similarity_path": str(similarity_path),
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "checkpoint_count": len(checkpoint_records),
        "test_event_count": len(splits),
        "prediction_hours_per_event": (
            prediction_hours[0]
            if len(set(prediction_hours)) == 1
            else prediction_hours
        ),
        "target_labels_used_for_training": False,
        "target_labels_used_for_similarity": False,
        "source_training_label": source_label,
        "water_level_reconstruction": reconstruct_water_level,
        "geographic_distance_used_in_primary_similarity": False,
        "checkpoints": checkpoint_records,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
