"""Source-only shrinkage of similarity weights for the seven-station panel
(parallel to five_station_shrinkage_runner.py, pointed at
seven_station_contract / seven_station_runner)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.exhaustive_contract import EXPECTED_HOURS, SEEDS
from src.experiments.seven_station_contract import STATIONS
from src.experiments.seven_station_runner import (
    calibration_keys,
    compose_method_forecasts,
    method_prediction_path,
    raw_prediction_path,
    selection_path,
    sha256_file,
    validate_raw_prediction,
    verify_seven_station_prediction_seal,
)
from src.experiments.shrinkage_runner import select_shrinkage, shrink_weights


LAMBDAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
SHRINKAGE_METHODS: tuple[str, ...] = (
    "nearest_source",
    "equal_ensemble",
    "similarity_top1",
    "similarity_weighted",
    "similarity_shrunk",
)
SHRINKAGE_ARTIFACT_COUNTS = {
    "parent_manifest": 1,
    "selection": 56,
    "method_prediction": 168,
    "shrinkage_audit": 168,
}


def shrinkage_selection_path(output_dir: Path, task: dict) -> Path:
    """Return the deterministic per-task shrinkage selection path."""
    return (
        Path(output_dir)
        / "calibration"
        / "selections"
        / f"{task['task_id']}.json"
    )


def _ordered_pair_information(
    pair_distances: pd.DataFrame,
    *,
    event_id: str,
    pseudo_target: str,
    candidate_source_ids: list[str],
) -> pd.DataFrame:
    subset = pair_distances.loc[
        (pair_distances["event_id"].astype(str) == str(event_id))
        & (pair_distances["target_station_id"].astype(str) == str(pseudo_target))
    ].copy()
    subset["source_station_id"] = subset["source_station_id"].astype(str)
    subset = subset.set_index("source_station_id").reindex(candidate_source_ids)
    if subset.isna().any().any():
        raise ValueError("parent pair-distance audit is incomplete")
    return subset.reset_index()


def calibrate_task_shrinkage(
    config: dict,
    data: pd.DataFrame,
    parent_dir: Path,
    output_dir: Path,
    *,
    task: dict,
) -> Path:
    """Select lambda on the exact source-only calibration grid."""
    parent_dir = Path(parent_dir)
    output_dir = Path(output_dir)
    verify_seven_station_prediction_seal(parent_dir / "prediction_manifest.json")
    output_path = shrinkage_selection_path(output_dir, task)
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            existing.get("target_station_id") == str(task["target_station_id"])
            and existing.get("test_event") == str(task["test_event"])
            and existing.get("lambda") in LAMBDAS
        ):
            return output_path
        raise ValueError(f"existing shrinkage selection conflicts: {output_path}")

    parent_selection_path = selection_path(parent_dir, task)
    parent_selection = json.loads(
        parent_selection_path.read_text(encoding="utf-8")
    )
    if parent_selection.get("target_labels_used") is not False:
        raise ValueError("parent similarity selection is not target-blind")
    if parent_selection.get("test_event_labels_used") is not False:
        raise ValueError("parent similarity selection used test-event labels")
    pair_distances = pd.read_csv(
        parent_selection["pair_distances_path"],
        dtype={
            "source_station_id": str,
            "target_station_id": str,
            "event_id": str,
        },
    )
    source_pool = [str(value) for value in task["calibration_station_ids"]]
    calibration_events = [str(value) for value in task["calibration_events"]]
    group_weights = (
        float(parent_selection["w_atmosphere"]),
        float(parent_selection["w_wave"]),
        float(parent_selection["w_coastal"]),
    )
    temperature = float(parent_selection["temperature"])

    calibration_items = []
    for key in calibration_keys(task, SEEDS):
        pseudo_target = str(key["pseudo_target_station_id"])
        event_id = str(key["event_id"])
        candidate_source_ids = [str(value) for value in key["candidate_source_ids"]]
        raw_path = raw_prediction_path(
            parent_dir,
            target_station_id=pseudo_target,
            event_id=event_id,
            seed=int(key["seed"]),
        )
        raw = pd.read_csv(raw_path, parse_dates=["timestamp_utc"])
        validate_raw_prediction(
            raw,
            source_station_ids=[
                station for station in STATIONS if station != pseudo_target
            ],
            target_station_id=pseudo_target,
        )
        predictions = raw[["timestamp_utc", *candidate_source_ids]].copy()
        labels = data.loc[
            (data["station_id"].astype(str) == pseudo_target)
            & (data["event_id"].astype(str) == event_id),
            ["timestamp_utc", "water_level_m"],
        ].sort_values("timestamp_utc")
        paired = predictions[["timestamp_utc"]].merge(
            labels,
            on="timestamp_utc",
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != EXPECTED_HOURS:
            raise ValueError("shrinkage calibration label alignment failed")
        observed = paired["water_level_m"].to_numpy(float)
        scale = float(np.std(observed))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("shrinkage calibration target has zero variance")
        information = _ordered_pair_information(
            pair_distances,
            event_id=event_id,
            pseudo_target=pseudo_target,
            candidate_source_ids=candidate_source_ids,
        )
        _, audit = compose_method_forecasts(
            predictions,
            information,
            group_weights=group_weights,
            temperature=temperature,
        )
        similarity_weights = np.array(
            [
                audit["source_weights"][source_id]
                for source_id in candidate_source_ids
            ],
            dtype=float,
        )
        calibration_items.append(
            {
                "predictions": predictions[candidate_source_ids].to_numpy(float),
                "observed": observed,
                "scale": scale,
                "similarity_weights": similarity_weights,
            }
        )

    candidate_rows = []
    for lambda_value in LAMBDAS:
        normalized_errors = []
        for item in calibration_items:
            weights = shrink_weights(
                item["similarity_weights"], lambda_value
            )
            forecast = item["predictions"] @ weights
            rmse = float(np.sqrt(np.mean((forecast - item["observed"]) ** 2)))
            normalized_errors.append(rmse / item["scale"])
        candidate_rows.append(
            {
                "lambda": lambda_value,
                "mean_nrmse": float(np.mean(normalized_errors)),
                "task_count": len(normalized_errors),
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    selected = select_shrinkage(candidates)
    candidate_path = (
        output_dir
        / "calibration"
        / "candidate_scores"
        / f"{task['task_id']}.csv"
    )
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidate_path, index=False)
    selection = {
        **selected,
        "target_station_id": str(task["target_station_id"]),
        "test_event": str(task["test_event"]),
        "calibration_station_ids": source_pool,
        "calibration_events": calibration_events,
        "calibration_task_count": len(calibration_items),
        "target_labels_used": False,
        "test_event_labels_used": False,
        "parent_selection_path": str(parent_selection_path),
        "candidate_scores_path": str(candidate_path),
        "parent_group_weights": {
            "atmosphere": group_weights[0],
            "wave": group_weights[1],
            "coastal": group_weights[2],
        },
        "parent_temperature": temperature,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def shrinkage_method_prediction_path(
    output_dir: Path,
    *,
    target_station_id: str,
    event_id: str,
    seed: int,
) -> Path:
    return (
        Path(output_dir)
        / "predictions"
        / "methods"
        / (
            f"target_{target_station_id}__event_{event_id}"
            f"__seed_{int(seed)}.csv"
        )
    )


def shrinkage_audit_path(
    output_dir: Path,
    *,
    target_station_id: str,
    event_id: str,
    seed: int,
) -> Path:
    return (
        Path(output_dir)
        / "predictions"
        / "shrinkage_audit"
        / (
            f"target_{target_station_id}__event_{event_id}"
            f"__seed_{int(seed)}.json"
        )
    )


def run_shrunk_prediction_task(
    parent_dir: Path,
    output_dir: Path,
    *,
    task: dict,
    seed: int,
    selection_path: Path,
) -> Path:
    """Apply one frozen lambda to the parent task's source forecasts."""
    parent_dir = Path(parent_dir)
    output_dir = Path(output_dir)
    target = str(task["target_station_id"])
    event = str(task["test_event"])
    source_ids = [str(value) for value in task["source_station_ids"]]
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if selection.get("target_station_id") != target:
        raise ValueError("shrinkage selection target mismatch")
    if selection.get("test_event") != event:
        raise ValueError("shrinkage selection event mismatch")
    if selection.get("target_labels_used") is not False:
        raise ValueError("shrinkage selection used target labels")
    if selection.get("test_event_labels_used") is not False:
        raise ValueError("shrinkage selection used test-event labels")
    lambda_value = float(selection["lambda"])

    raw_path = raw_prediction_path(
        parent_dir,
        target_station_id=target,
        event_id=event,
        seed=int(seed),
    )
    raw = pd.read_csv(raw_path, parse_dates=["timestamp_utc"])
    validate_raw_prediction(
        raw,
        source_station_ids=source_ids,
        target_station_id=target,
    )
    parent_method_path = method_prediction_path(
        parent_dir,
        target_station_id=target,
        event_id=event,
        seed=int(seed),
    )
    parent_methods = pd.read_csv(
        parent_method_path, parse_dates=["timestamp_utc"]
    )
    parent_audit_path = (
        parent_dir
        / "predictions"
        / "similarity_audit"
        / f"target_{target}__event_{event}__seed_{int(seed)}.json"
    )
    parent_audit = json.loads(parent_audit_path.read_text(encoding="utf-8"))
    similarity_weights = np.array(
        [parent_audit["source_weights"][source_id] for source_id in source_ids],
        dtype=float,
    )
    shrunk = shrink_weights(similarity_weights, lambda_value)
    values = raw[source_ids].to_numpy(float)
    forecast = values @ shrunk

    result = parent_methods.copy()
    if result["timestamp_utc"].nunique() != EXPECTED_HOURS:
        raise ValueError("parent method prediction timestamps are incomplete")
    result["similarity_shrunk"] = forecast
    expected_columns = ["timestamp_utc", *SHRINKAGE_METHODS]
    if result.columns.tolist() != expected_columns:
        raise ValueError("shrinkage method columns do not match contract")
    output_path = shrinkage_method_prediction_path(
        output_dir,
        target_station_id=target,
        event_id=event,
        seed=int(seed),
    )
    audit_path = shrinkage_audit_path(
        output_dir,
        target_station_id=target,
        event_id=event,
        seed=int(seed),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    audit_path.write_text(
        json.dumps(
            {
                "target_station_id": target,
                "event_id": event,
                "seed": int(seed),
                "lambda": lambda_value,
                "source_station_ids": source_ids,
                "parent_similarity_weights": {
                    source_id: float(weight)
                    for source_id, weight in zip(
                        source_ids, similarity_weights, strict=True
                    )
                },
                "shrunk_source_weights": {
                    source_id: float(weight)
                    for source_id, weight in zip(
                        source_ids, shrunk, strict=True
                    )
                },
                "target_labels_used": False,
                "test_event_labels_used": False,
                "parent_raw_prediction_path": str(raw_path),
                "parent_method_prediction_path": str(parent_method_path),
                "parent_similarity_audit_path": str(parent_audit_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def write_shrinkage_prediction_seal(
    output_path: Path,
    *,
    artifacts: dict[str, list[Path]],
    metadata: dict,
    expected_counts: dict[str, int] | None = None,
) -> Path:
    """Hash the exact shrinkage prediction grid before label reveal."""
    output_path = Path(output_path)
    counts = dict(
        SHRINKAGE_ARTIFACT_COUNTS
        if expected_counts is None
        else expected_counts
    )
    if set(artifacts) != set(counts):
        raise ValueError("shrinkage seal artifact roles mismatch")
    for key in (
        "active_target_labels_used_for_calibration",
        "active_target_labels_used_for_prediction",
    ):
        if metadata.get(key) is not False:
            raise ValueError(f"{key} must be false before sealing")
    records: dict[str, list[dict]] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for role, expected_count in counts.items():
        paths = [Path(path).resolve() for path in artifacts[role]]
        if len(paths) != int(expected_count):
            raise ValueError(f"shrinkage seal {role} count mismatch")
        role_records = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            role_records.append(
                {
                    "path": str(
                        Path(
                            os.path.relpath(
                                path, output_path.parent.resolve()
                            )
                        )
                    ),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
        records[role] = role_records
    manifest = {
        **metadata,
        "labels_revealed": False,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "expected_artifact_counts": counts,
        "artifacts": records,
    }
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def verify_shrinkage_prediction_seal(
    manifest_path: Path,
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    """Verify exact counts, sizes, and hashes for a shrinkage prediction seal."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("labels_revealed") is not False:
        raise ValueError("shrinkage prediction seal is already revealed")
    counts = dict(
        SHRINKAGE_ARTIFACT_COUNTS
        if expected_counts is None
        else expected_counts
    )
    if manifest.get("expected_artifact_counts") != counts:
        raise ValueError("shrinkage prediction seal count contract mismatch")
    if set(manifest.get("artifacts", {})) != set(counts):
        raise ValueError("shrinkage prediction seal artifact roles mismatch")
    for role, expected_count in counts.items():
        records = manifest["artifacts"][role]
        if len(records) != int(expected_count):
            raise ValueError(f"shrinkage prediction seal {role} count mismatch")
        for record in records:
            path = (manifest_path.parent / record["path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"size mismatch for sealed artifact: {path}")
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"hash mismatch for sealed artifact: {path}")
    return manifest
