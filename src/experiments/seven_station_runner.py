"""Resumable training, transfer prediction, and calibration for the
seven-station 56-task grid (parallel to five_station_runner.py, adding
Sewells Point (semi_enclosed) and Money Point (enclosed) as two
non-open-exposure candidate sources -- a deliberate stress test of whether
learned weighting can identify and down-weight genuinely anomalous
candidate sources, something equal-weight ensembling cannot do)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.experiments.exhaustive_contract import EVENTS, EXPECTED_HOURS, build_event_split
from src.experiments.seven_station_contract import STATIONS, validate_seven_station_config
from src.experiments.residual_forecasting import (
    build_target_inputs,
    predict_residual_checkpoint,
    train_residual_checkpoint,
)
from src.experiments.closed_book_seal import (
    apply_frozen_similarity_weights,
    sha256_file,
)
from src.experiments.source_only_calibration import (
    build_pair_distances,
    fit_descriptor_reference,
    select_calibration,
    simplex_grid,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ARTIFACT_COUNTS = {
    "config": 1,
    "dataset": 1,
    "coastal_context": 1,
    "checkpoint": 168,
    "raw_prediction": 168,
    "method_prediction": 168,
    "calibration_selection": 56,
    "similarity_audit": 168,
}


def load_seven_station_data(
    config_path: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load and strictly validate the frozen seven-station event grid."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_seven_station_config(config)
    data_path = ROOT / config["data"]["event_dataset"]
    coastal_path = ROOT / config["data"]["coastal_context"]
    data = pd.read_csv(
        data_path,
        dtype={"station_id": str, "event_id": str},
        parse_dates=["timestamp_utc"],
    )
    data = data.loc[
        data["station_id"].isin(STATIONS) & data["event_id"].isin(EVENTS)
    ].copy()
    required = {
        "station_id",
        "event_id",
        "timestamp_utc",
        "latitude",
        "longitude",
        "water_level_m",
        "ntr_m",
        *config["features"]["dynamic"],
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"event dataset missing columns: {missing}")
    expected_rows = len(STATIONS) * len(EVENTS) * EXPECTED_HOURS
    if len(data) != expected_rows:
        raise ValueError(f"frozen event grid requires {expected_rows} rows")
    counts = data.groupby(["station_id", "event_id"]).size()
    if len(counts) != len(STATIONS) * len(EVENTS) or not counts.eq(
        EXPECTED_HOURS
    ).all():
        raise ValueError("every frozen station-event pair requires 216 rows")
    if data.duplicated(["station_id", "event_id", "timestamp_utc"]).any():
        raise ValueError("event dataset contains duplicate UTC timestamps")
    finite_columns = [
        "latitude",
        "longitude",
        "water_level_m",
        "ntr_m",
        *config["features"]["dynamic"],
    ]
    values = data[finite_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("frozen event grid contains missing/non-finite values")
    data = data.sort_values(
        ["station_id", "event_id", "timestamp_utc"]
    ).reset_index(drop=True)

    coastal = pd.read_csv(coastal_path, dtype={"station_id": str})
    coastal = coastal.loc[coastal["station_id"].isin(STATIONS)].copy()
    coastal_required = {"station_id", *config["similarity"]["groups"]["coastal"]}
    missing_coastal = sorted(coastal_required - set(coastal))
    if missing_coastal:
        raise ValueError(f"coastal context missing columns: {missing_coastal}")
    if set(coastal["station_id"]) != set(STATIONS):
        raise ValueError("coastal context does not cover all seven stations")
    coastal_values = coastal[config["similarity"]["groups"]["coastal"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(coastal_values.to_numpy(float)).all():
        raise ValueError("coastal context contains missing/non-finite values")
    return config, data, coastal


def load_full_coastal_context(config: dict) -> pd.DataFrame:
    """Load every built station's static coastal descriptors, unfiltered.

    Used only to widen the coastal-group standardization reference beyond
    the seven active stations; it never touches ``water_level_m``/``ntr_m``.
    """
    coastal_path = ROOT / config["data"]["coastal_context"]
    coastal = pd.read_csv(coastal_path, dtype={"station_id": str})
    coastal_required = {"station_id", *config["similarity"]["groups"]["coastal"]}
    missing_coastal = sorted(coastal_required - set(coastal))
    if missing_coastal:
        raise ValueError(f"coastal context missing columns: {missing_coastal}")
    coastal_values = coastal[config["similarity"]["groups"]["coastal"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(coastal_values.to_numpy(float)).all():
        raise ValueError("coastal context contains missing/non-finite values")
    if not set(STATIONS).issubset(set(coastal["station_id"])):
        raise ValueError("full coastal context does not cover the active panel")
    return coastal


def checkpoint_path(output_dir: Path, job: dict) -> Path:
    """Return one deterministic reusable checkpoint path."""
    return (
        Path(output_dir)
        / "checkpoints"
        / str(job["test_event"])
        / (
            f"source_{job['source_station_id']}"
            f"__seed_{int(job['seed'])}.pt"
        )
    )


def raw_prediction_path(
    output_dir: Path,
    *,
    target_station_id: str,
    event_id: str,
    seed: int,
) -> Path:
    """Return one deterministic target-event-seed raw prediction path."""
    return (
        Path(output_dir)
        / "predictions"
        / "raw_sources"
        / f"target_{target_station_id}__event_{event_id}__seed_{int(seed)}.csv"
    )


def method_prediction_path(
    output_dir: Path,
    *,
    target_station_id: str,
    event_id: str,
    seed: int,
) -> Path:
    """Return one deterministic method-prediction path."""
    return (
        Path(output_dir)
        / "predictions"
        / "methods"
        / f"target_{target_station_id}__event_{event_id}__seed_{int(seed)}.csv"
    )


def selection_path(output_dir: Path, task: dict) -> Path:
    """Return one deterministic target-event calibration selection path."""
    return (
        Path(output_dir)
        / "calibration"
        / "selections"
        / (
            f"target_{task['target_station_id']}"
            f"__event_{task['test_event']}.json"
        )
    )


def checkpoint_matches_job(
    path: Path,
    job: dict,
    features: list[str],
) -> bool:
    """Return true only for a complete checkpoint with exact frozen metadata."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError):
        return False
    expected = {
        "source_station_id": str(job["source_station_id"]),
        "test_event": str(job["test_event"]),
        "train_events": list(job["train_events"]),
        "validation_event": str(job["validation_event"]),
        "seed": int(job["seed"]),
        "features": list(features),
        "source_training_label": "ntr_m",
    }
    return all(payload.get(key) == value for key, value in expected.items())


def validate_raw_prediction(
    frame: pd.DataFrame,
    *,
    source_station_ids: list[str],
    target_station_id: str,
) -> None:
    """Validate a target-blind table of six source-model predictions."""
    expected_columns = ["timestamp_utc", *source_station_ids]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(
            f"raw prediction columns must be exactly {expected_columns}"
        )
    if str(target_station_id) in frame.columns:
        raise ValueError("target station model is forbidden in its own task")
    if len(frame) != EXPECTED_HOURS:
        raise ValueError("raw prediction requires exactly 216 rows")
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if timestamps.nunique() != EXPECTED_HOURS:
        raise ValueError("raw prediction requires 216 unique UTC timestamps")
    values = frame[source_station_ids].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("raw source predictions must be finite")


def run_checkpoint_pool(
    config: dict,
    data: pd.DataFrame,
    output_dir: Path,
    *,
    jobs: list[dict],
    max_epochs_override: int | None = None,
) -> list[dict]:
    """Train missing checkpoints and safely resume exact completed jobs."""
    validate_seven_station_config(config)
    features = list(config["features"]["dynamic"])
    records = []
    for job in jobs:
        path = checkpoint_path(output_dir, job)
        history_path = (
            Path(output_dir)
            / "training_history"
            / str(job["test_event"])
            / (
                f"source_{job['source_station_id']}"
                f"__seed_{int(job['seed'])}.csv"
            )
        )
        if checkpoint_matches_job(path, job, features):
            records.append(
                {
                    **job,
                    "status": "reused",
                    "checkpoint_path": str(path),
                }
            )
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        training = train_residual_checkpoint(
            data,
            source_station_id=str(job["source_station_id"]),
            train_events=list(job["train_events"]),
            validation_event=str(job["validation_event"]),
            test_event=str(job["test_event"]),
            features=features,
            model_config=dict(config["model"]),
            seed=int(job["seed"]),
            checkpoint_path=path,
            history_path=history_path,
            max_epochs_override=max_epochs_override,
        )
        if not checkpoint_matches_job(path, job, features):
            raise RuntimeError(f"trained checkpoint failed metadata audit: {path}")
        records.append(
            {
                **job,
                **training,
                "status": "trained",
                "checkpoint_path": str(path),
                "history_path": str(history_path),
            }
        )
    return records


def run_raw_prediction_task(
    config: dict,
    data: pd.DataFrame,
    output_dir: Path,
    *,
    task: dict,
    seed: int,
) -> Path:
    """Apply six non-target checkpoints to one target event without labels."""
    validate_seven_station_config(config)
    source_station_ids = [str(value) for value in task["source_station_ids"]]
    target_station_id = str(task["target_station_id"])
    event_id = str(task["test_event"])
    output_path = raw_prediction_path(
        output_dir,
        target_station_id=target_station_id,
        event_id=event_id,
        seed=int(seed),
    )
    if output_path.is_file():
        existing = pd.read_csv(output_path)
        validate_raw_prediction(
            existing,
            source_station_ids=source_station_ids,
            target_station_id=target_station_id,
        )
        return output_path

    features = list(config["features"]["dynamic"])
    target_rows = data.loc[
        (data["station_id"] == target_station_id)
        & (data["event_id"] == event_id)
    ].sort_values("timestamp_utc")
    target_inputs = build_target_inputs(target_rows, features)
    split = build_event_split(event_id)
    predictions: pd.DataFrame | None = None
    for source_station_id in source_station_ids:
        job = {
            "test_event": event_id,
            "source_station_id": source_station_id,
            "seed": int(seed),
            "validation_event": split["validation_event"],
            "train_events": split["train_events"],
        }
        path = checkpoint_path(output_dir, job)
        if not checkpoint_matches_job(path, job, features):
            raise FileNotFoundError(f"valid checkpoint missing for prediction: {path}")
        forecast = predict_residual_checkpoint(path, target_inputs)
        if predictions is None:
            predictions = forecast[["timestamp_utc"]].copy()
        predictions[source_station_id] = forecast[
            "predicted_water_level_m"
        ].to_numpy(float)
    if predictions is None:
        raise RuntimeError("no source forecast was produced")
    validate_raw_prediction(
        predictions,
        source_station_ids=source_station_ids,
        target_station_id=target_station_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return output_path


def calibration_keys(task: dict, seeds: list[int]) -> list[dict]:
    """Enumerate source-only pseudo-target tasks for one active target fold."""
    source_pool = [str(value) for value in task["calibration_station_ids"]]
    active_target = str(task["target_station_id"])
    test_event = str(task["test_event"])
    if active_target in source_pool:
        raise ValueError("active target is forbidden in calibration station pool")
    if test_event in task["calibration_events"]:
        raise ValueError("active test event is forbidden in calibration events")
    keys = []
    for event_id in task["calibration_events"]:
        for pseudo_target in source_pool:
            candidates = [
                source_id for source_id in source_pool if source_id != pseudo_target
            ]
            for seed in seeds:
                keys.append(
                    {
                        "event_id": str(event_id),
                        "pseudo_target_station_id": pseudo_target,
                        "candidate_source_ids": candidates,
                        "seed": int(seed),
                    }
                )
    return keys


def compose_method_forecasts(
    predictions: pd.DataFrame,
    information: pd.DataFrame,
    *,
    group_weights: tuple[float, float, float],
    temperature: float,
) -> tuple[dict[str, np.ndarray], dict]:
    """Compose four methods from one identical source-forecast pool."""
    source_ids = information["source_station_id"].astype(str).tolist()
    if predictions.columns.tolist() != ["timestamp_utc", *source_ids]:
        raise ValueError("source prediction columns do not match pair information")
    weights = apply_frozen_similarity_weights(
        information,
        group_weights=group_weights,
        temperature=temperature,
    ).reindex(source_ids)
    values = predictions[source_ids].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("source predictions must be finite")
    nearest_id = str(
        information.loc[
            information["geo_distance_km"].astype(float).idxmin(),
            "source_station_id",
        ]
    )
    top1_id = str(weights.idxmax())
    methods = {
        "nearest_source": predictions[nearest_id].to_numpy(float),
        "equal_ensemble": values.mean(axis=1),
        "similarity_top1": predictions[top1_id].to_numpy(float),
        "similarity_weighted": values @ weights.to_numpy(float),
    }
    audit = {
        "nearest_source_station_id": nearest_id,
        "top1_source_station_id": top1_id,
        "source_weights": {
            source_id: float(weights.loc[source_id]) for source_id in source_ids
        },
    }
    return methods, audit


def _raw_sources_for_target(target_station_id: str) -> list[str]:
    return [
        station_id for station_id in STATIONS if station_id != str(target_station_id)
    ]


def _coastal_reference_station_ids(
    coastal: pd.DataFrame, *, active_target: str
) -> list[str]:
    return sorted(
        str(station)
        for station in coastal["station_id"].unique()
        if str(station) != str(active_target)
    )


def calibrate_task_similarity(
    config: dict,
    data: pd.DataFrame,
    coastal: pd.DataFrame,
    output_dir: Path,
    *,
    task: dict,
) -> Path:
    """Calibrate weights without the active target station or test event.

    ``coastal`` should be the full built-station inventory (see
    ``load_full_coastal_context``), not the 7-station frame returned by
    ``load_seven_station_data`` -- the coastal-group standardization
    reference is fit from every other built station, not just the six
    active sources.
    """
    validate_seven_station_config(config)
    output_path = selection_path(output_dir, task)
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            existing.get("target_station_id") == str(task["target_station_id"])
            and existing.get("test_event") == str(task["test_event"])
            and str(task["target_station_id"])
            not in existing.get("calibration_station_ids", [])
            and str(task["test_event"])
            not in existing.get("calibration_events", [])
        ):
            return output_path
        raise ValueError(f"existing calibration selection conflicts: {output_path}")

    seeds = [int(value) for value in config["model"]["random_seeds"]]
    source_pool = [str(value) for value in task["calibration_station_ids"]]
    calibration_events = [str(value) for value in task["calibration_events"]]
    group_variables = {
        name: list(columns)
        for name, columns in config["similarity"]["groups"].items()
    }
    coastal_reference_station_ids = _coastal_reference_station_ids(
        coastal, active_target=str(task["target_station_id"])
    )
    reference = fit_descriptor_reference(
        data,
        coastal,
        source_station_ids=source_pool,
        calibration_events=calibration_events,
        group_variables=group_variables,
        coastal_reference_station_ids=coastal_reference_station_ids,
    )
    pair_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    pair_frames = []
    for event_id in calibration_events:
        for pseudo_target in source_pool:
            candidates = [
                source_id for source_id in source_pool if source_id != pseudo_target
            ]
            information = build_pair_distances(
                data,
                coastal,
                candidate_source_ids=candidates,
                target_station_id=pseudo_target,
                event_id=event_id,
                group_variables=group_variables,
                reference=reference,
            )
            pair_lookup[(event_id, pseudo_target)] = information
            pair_frames.append(information)

    calibration_items = []
    for key in calibration_keys(task, seeds):
        raw_path = raw_prediction_path(
            output_dir,
            target_station_id=key["pseudo_target_station_id"],
            event_id=key["event_id"],
            seed=key["seed"],
        )
        raw = pd.read_csv(raw_path, parse_dates=["timestamp_utc"])
        raw_sources = _raw_sources_for_target(key["pseudo_target_station_id"])
        validate_raw_prediction(
            raw,
            source_station_ids=raw_sources,
            target_station_id=key["pseudo_target_station_id"],
        )
        predictions = raw[
            ["timestamp_utc", *key["candidate_source_ids"]]
        ].copy()
        labels = data.loc[
            (data["station_id"] == key["pseudo_target_station_id"])
            & (data["event_id"] == key["event_id"]),
            ["timestamp_utc", "water_level_m"],
        ].sort_values("timestamp_utc")
        merged = predictions[["timestamp_utc"]].merge(
            labels,
            on="timestamp_utc",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != EXPECTED_HOURS:
            raise ValueError("source-only calibration label alignment failed")
        scale = float(np.std(merged["water_level_m"].to_numpy(float)))
        if scale <= 0:
            raise ValueError("source-only calibration target has zero variance")
        calibration_items.append(
            {
                "key": key,
                "predictions": predictions,
                "observed": merged["water_level_m"].to_numpy(float),
                "scale": scale,
                "information": pair_lookup[
                    (key["event_id"], key["pseudo_target_station_id"])
                ],
            }
        )

    candidate_rows = []
    for group_weights in simplex_grid(float(config["similarity"]["grid_step"])):
        for temperature in [
            float(value) for value in config["similarity"]["temperatures"]
        ]:
            normalized_errors = []
            for item in calibration_items:
                methods, _ = compose_method_forecasts(
                    item["predictions"],
                    item["information"],
                    group_weights=group_weights,
                    temperature=temperature,
                )
                residual = (
                    methods["similarity_weighted"] - item["observed"]
                )
                rmse = float(np.sqrt(np.mean(residual**2)))
                normalized_errors.append(rmse / item["scale"])
            candidate_rows.append(
                {
                    "w_atmosphere": group_weights[0],
                    "w_wave": group_weights[1],
                    "w_coastal": group_weights[2],
                    "temperature": temperature,
                    "mean_nrmse": float(np.mean(normalized_errors)),
                    "task_count": len(normalized_errors),
                }
            )
    candidates = pd.DataFrame(candidate_rows)
    selected = select_calibration(candidates)

    target_dir = output_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem
    candidate_path = (
        Path(output_dir) / "calibration" / "candidate_scores" / f"{stem}.csv"
    )
    pair_path = (
        Path(output_dir) / "calibration" / "pair_distances" / f"{stem}.csv"
    )
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidate_path, index=False)
    pd.concat(pair_frames, ignore_index=True).to_csv(pair_path, index=False)
    selection = {
        **selected,
        "target_station_id": str(task["target_station_id"]),
        "test_event": str(task["test_event"]),
        "calibration_station_ids": source_pool,
        "calibration_events": calibration_events,
        "calibration_task_count": len(calibration_items),
        "coastal_reference_station_ids": coastal_reference_station_ids,
        "target_labels_used": False,
        "test_event_labels_used": False,
        "descriptor_reference": reference,
        "candidate_scores_path": str(candidate_path),
        "pair_distances_path": str(pair_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def run_method_prediction_task(
    config: dict,
    data: pd.DataFrame,
    coastal: pd.DataFrame,
    output_dir: Path,
    *,
    task: dict,
    seed: int,
    selection_path: Path,
) -> Path:
    """Apply one source-only selection to an active target's raw forecasts."""
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    target_station_id = str(task["target_station_id"])
    event_id = str(task["test_event"])
    if selection.get("target_station_id") != target_station_id:
        raise ValueError("calibration target does not match prediction task")
    if selection.get("test_event") != event_id:
        raise ValueError("calibration event does not match prediction task")
    if selection.get("target_labels_used") is not False:
        raise ValueError("calibration used active target labels")
    source_ids = [str(value) for value in task["source_station_ids"]]
    raw_path = raw_prediction_path(
        output_dir,
        target_station_id=target_station_id,
        event_id=event_id,
        seed=int(seed),
    )
    raw = pd.read_csv(raw_path, parse_dates=["timestamp_utc"])
    validate_raw_prediction(
        raw,
        source_station_ids=source_ids,
        target_station_id=target_station_id,
    )
    group_variables = {
        name: list(columns)
        for name, columns in config["similarity"]["groups"].items()
    }
    information = build_pair_distances(
        data,
        coastal,
        candidate_source_ids=source_ids,
        target_station_id=target_station_id,
        event_id=event_id,
        group_variables=group_variables,
        reference=selection["descriptor_reference"],
    )
    group_weights = (
        float(selection["w_atmosphere"]),
        float(selection["w_wave"]),
        float(selection["w_coastal"]),
    )
    methods, audit = compose_method_forecasts(
        raw,
        information,
        group_weights=group_weights,
        temperature=float(selection["temperature"]),
    )
    result = raw[["timestamp_utc"]].copy()
    for method, values in methods.items():
        result[method] = values
    output_path = method_prediction_path(
        output_dir,
        target_station_id=target_station_id,
        event_id=event_id,
        seed=int(seed),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    audit_path = (
        Path(output_dir)
        / "predictions"
        / "similarity_audit"
        / f"target_{target_station_id}__event_{event_id}__seed_{int(seed)}.json"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "target_station_id": target_station_id,
                "event_id": event_id,
                "seed": int(seed),
                "group_weights": group_weights,
                "temperature": float(selection["temperature"]),
                **audit,
                "pair_distances": information.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def write_seven_station_prediction_seal(
    output_path: Path,
    *,
    artifacts: dict[str, list[Path]],
    metadata: dict,
    expected_counts: dict[str, int] | None = None,
) -> Path:
    """Hash every seven-station prediction artifact before evaluation."""
    output_path = Path(output_path)
    counts = dict(EXPECTED_ARTIFACT_COUNTS if expected_counts is None else expected_counts)
    if set(artifacts) != set(counts):
        raise ValueError("seven-station seal artifact roles do not match contract")
    for key in (
        "active_target_labels_used_for_training",
        "active_target_labels_used_for_calibration",
        "active_target_labels_used_for_prediction",
    ):
        if metadata.get(key) is not False:
            raise ValueError(f"{key} must be explicitly false")
    records = {}
    for role, expected_count in counts.items():
        paths = [Path(path) for path in artifacts[role]]
        if len(paths) != int(expected_count):
            raise ValueError(
                f"{role} requires {expected_count} files, got {len(paths)}"
            )
        records[role] = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            records[role].append(
                {
                    "path": Path(
                        os.path.relpath(path.resolve(), output_path.parent.resolve())
                    ).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        **metadata,
        "labels_revealed": False,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "expected_artifact_counts": counts,
        "artifacts": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def verify_seven_station_prediction_seal(
    manifest_path: Path,
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    """Verify unrevealed state, exact grid counts, sizes, and SHA-256 hashes."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("labels_revealed") is not False:
        raise ValueError("prediction seal is already revealed or invalid")
    counts = dict(EXPECTED_ARTIFACT_COUNTS if expected_counts is None else expected_counts)
    if manifest.get("expected_artifact_counts") != counts:
        raise ValueError("prediction seal expected counts mismatch")
    if set(manifest.get("artifacts", {})) != set(counts):
        raise ValueError("prediction seal artifact roles mismatch")
    for role, expected_count in counts.items():
        records = manifest["artifacts"][role]
        if len(records) != int(expected_count):
            raise ValueError(f"prediction seal {role} count mismatch")
        for record in records:
            path = (manifest_path.parent / record["path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"size mismatch for sealed artifact: {path}")
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"hash mismatch for sealed artifact: {path}")
    return manifest
