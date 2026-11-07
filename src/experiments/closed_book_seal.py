"""Hash seal and frozen-weight helpers for the Beaufort closed-book run."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_ARTIFACT_ROLES = {
    "config",
    "calibration",
    "checkpoint",
    "raw_prediction",
    "method_prediction",
}
SIMILARITY_GROUPS = ("atmosphere", "wave", "coastal")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stored_path(path: Path, manifest_path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), manifest_path.parent.resolve())).as_posix()


def write_prediction_seal(
    output_path: Path,
    *,
    artifacts: dict[str, list[Path]],
    metadata: dict,
) -> Path:
    """Write an unrevealed manifest that seals all prediction inputs/outputs."""
    output_path = Path(output_path)
    missing_roles = sorted(REQUIRED_ARTIFACT_ROLES - set(artifacts))
    if missing_roles:
        raise ValueError(f"missing artifact roles: {missing_roles}")
    if metadata.get("beaufort_labels_used_for_training") is not False:
        raise ValueError("training label-use flag must be explicitly false")
    if metadata.get("beaufort_labels_used_for_calibration") is not False:
        raise ValueError("calibration label-use flag must be explicitly false")

    records: dict[str, list[dict]] = {}
    for role in sorted(artifacts):
        paths = [Path(path) for path in artifacts[role]]
        if not paths:
            raise ValueError(f"artifact role has no files: {role}")
        records[role] = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            records[role].append(
                {
                    "path": _stored_path(path, output_path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    manifest = {
        **metadata,
        "labels_revealed": False,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def verify_prediction_seal(manifest_path: Path) -> dict:
    """Verify unrevealed state and every file digest in a prediction seal."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("labels_revealed") is not False:
        raise ValueError("prediction seal is already revealed or invalid")
    missing_roles = sorted(
        REQUIRED_ARTIFACT_ROLES - set(manifest.get("artifacts", {}))
    )
    if missing_roles:
        raise ValueError(f"missing artifact roles: {missing_roles}")
    for role, records in manifest["artifacts"].items():
        if not records:
            raise ValueError(f"artifact role has no files: {role}")
        for record in records:
            path = (manifest_path.parent / record["path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"size mismatch for sealed artifact: {path}")
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"hash mismatch for sealed artifact: {path}")
    return manifest


def apply_frozen_similarity_weights(
    source_information: pd.DataFrame,
    *,
    group_weights: tuple[float, float, float],
    temperature: float,
) -> pd.Series:
    """Apply source-calibrated weights to a new target without target labels."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if any(value < 0 for value in group_weights) or not np.isclose(
        sum(group_weights), 1.0
    ):
        raise ValueError("group weights must be nonnegative and sum to one")
    columns = [f"{group}_distance" for group in SIMILARITY_GROUPS]
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


def combine_closed_book_forecasts(
    source_predictions: pd.DataFrame,
    source_information: pd.DataFrame,
    *,
    group_weights: tuple[float, float, float],
    temperature: float,
) -> tuple[dict[str, np.ndarray], dict]:
    """Create the four preregistered methods from one shared source pool."""
    source_ids = source_information["source_station_id"].astype(str).tolist()
    missing = sorted(set(source_ids) - set(source_predictions))
    if missing:
        raise ValueError(f"missing source prediction columns: {missing}")
    weights = apply_frozen_similarity_weights(
        source_information,
        group_weights=group_weights,
        temperature=temperature,
    ).reindex(source_ids)
    values = source_predictions[source_ids].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("source predictions must be finite")
    nearest_id = str(
        source_information.loc[
            source_information["geo_distance_km"].astype(float).idxmin(),
            "source_station_id",
        ]
    )
    top1_id = str(weights.idxmax())
    methods = {
        "nearest_source": source_predictions[nearest_id].to_numpy(float),
        "equal_ensemble": values.mean(axis=1),
        "calibrated_similarity_top1": source_predictions[top1_id].to_numpy(float),
        "calibrated_similarity_weighted": values @ weights.to_numpy(float),
    }
    audit = {
        "nearest_source_station_id": nearest_id,
        "top1_source_station_id": top1_id,
        "source_weights": {
            source_id: float(weights.loc[source_id]) for source_id in source_ids
        },
    }
    return methods, audit


def classify_confirmation(aggregated_metrics: pd.DataFrame) -> str:
    """Apply the preregistered two-event decision rule."""
    required_columns = {"event_id", "method", "rmse", "mae", "kge", "nse"}
    missing = sorted(required_columns - set(aggregated_metrics))
    if missing:
        raise ValueError(f"missing decision columns: {missing}")
    required_methods = {
        "nearest_source",
        "equal_ensemble",
        "calibrated_similarity_weighted",
    }
    event_tables: list[pd.DataFrame] = []
    for _, event in aggregated_metrics.groupby("event_id", sort=True):
        table = event.set_index("method")
        if not required_methods.issubset(table.index):
            raise ValueError("each event requires nearest, equal, and weighted metrics")
        event_tables.append(table)
    if len(event_tables) != 2:
        raise ValueError("confirmation decision requires exactly two events")

    confirmed_by_event = []
    beats_equal_rmse = []
    beats_nearest_rmse = []
    any_rmse_gain = False
    for table in event_tables:
        weighted = table.loc["calibrated_similarity_weighted"]
        nearest = table.loc["nearest_source"]
        equal = table.loc["equal_ensemble"]
        lower_rmse = (
            float(weighted["rmse"]) < float(nearest["rmse"])
            and float(weighted["rmse"]) < float(equal["rmse"])
        )
        nonworse_mae = (
            float(weighted["mae"]) <= float(nearest["mae"])
            and float(weighted["mae"]) <= float(equal["mae"])
        )
        better_skill = (
            float(weighted["kge"]) > max(float(nearest["kge"]), float(equal["kge"]))
            or float(weighted["nse"])
            > max(float(nearest["nse"]), float(equal["nse"]))
        )
        confirmed_by_event.append(lower_rmse and nonworse_mae and better_skill)
        beats_equal_rmse.append(float(weighted["rmse"]) < float(equal["rmse"]))
        beats_nearest_rmse.append(float(weighted["rmse"]) < float(nearest["rmse"]))
        any_rmse_gain = any_rmse_gain or lower_rmse
    if all(confirmed_by_event):
        return "CONFIRMED"
    if (
        all(beats_equal_rmse)
        or sum(beats_nearest_rmse) == 1
        or any_rmse_gain
    ):
        return "PARTIAL"
    return "NOT_CONFIRMED"
