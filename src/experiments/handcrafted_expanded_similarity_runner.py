"""Expanded hand-crafted (zero-learning, grid-search-only) similarity
calibration for the seven-station panel. Generalizes
seven_station_runner.calibrate_task_similarity/compose_method_forecasts
(hardcoded to exactly 3 groups: atmosphere/wave/coastal) to an arbitrary
number of groups, so additional static descriptors (water_fraction,
shelf_depth) and a source-history prior can be added to the same
brute-force simplex-grid-search mechanism without introducing any learned
(gradient-fit) free parameters. Reuses the frozen seven_station_56 raw
predictions and checkpoints without retraining -- only the similarity
calibration layer changes. group_weights/temperature/lambda are jointly
selected in one grid search (matching the Round 9 principle: never select
a calibration parameter against anything other than the real downstream
blended-prediction loss), unlike the original two-stage hand-crafted
pipeline (group_weights+temperature, then a separate shrinkage pass)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import src.experiments.seven_station_runner as seven_station_parent
from src.experiments.closed_book_seal import sha256_file
from src.experiments.exhaustive_contract import EVENTS, SEEDS
from src.experiments.expanded_coastal_similarity_contract import (
    compute_static_descriptor_distance,
)
from src.experiments.seven_station_contract import STATIONS, build_task_grid
from src.experiments.seven_station_runner import (
    load_seven_station_data,
    load_full_coastal_context,
    method_prediction_path,
    raw_prediction_path,
    validate_raw_prediction,
    verify_seven_station_prediction_seal,
)
from src.experiments.shrinkage_runner import shrink_weights
from src.experiments.source_only_calibration import (
    build_pair_distances,
    fit_descriptor_reference,
)
from src.metrics import evaluate_event


BASE_GROUP_VARIABLES = {
    "atmosphere": ["u10", "v10", "t2m", "msl", "tp"],
    "wave": ["sst", "swh", "mwp", "mwd_sin", "mwd_cos"],
    "coastal": [
        "coastline_orientation_sin2",
        "coastline_orientation_cos2",
        "nearshore_log_depth",
        "shelf_slope_m_per_km",
        "open_ray_fraction_50km",
    ],
}
DEFAULT_TEMPERATURES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
DEFAULT_LAMBDAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
EXTRA_STATIC_COLUMNS = {
    "water_fraction_distance": "water_fraction_50km",
    "shelf_depth_distance": "shelf_mean_depth_m",
}
EXPECTED_ARTIFACT_COUNTS = {
    "parent_manifest": 1,
    "selection": 56,
    "method_prediction": 168,
    "similarity_audit": 168,
}
METHOD_COLUMNS = (
    "timestamp_utc",
    "nearest_source",
    "equal_ensemble",
    "similarity_top1",
    "similarity_weighted",
    "similarity_shrunk",
)


@dataclass(frozen=True)
class StageConfig:
    name: str
    grid_step: float = 0.1
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS
    include_water_fraction: bool = False
    include_shelf_depth: bool = False
    include_source_prior: bool = False
    objective: str = "whole_series"  # "whole_series" | "top10pct"
    extra_group_names: tuple[str, ...] = field(default=())

    def group_names(self) -> tuple[str, ...]:
        names = ["atmosphere", "wave", "coastal"]
        if self.include_water_fraction:
            names.append("water_fraction")
        if self.include_shelf_depth:
            names.append("shelf_depth")
        if self.include_source_prior:
            names.append("source_prior")
        return tuple(names)


def simplex_grid_n(step: float, n: int) -> list[tuple[float, ...]]:
    """Deterministic n-component simplex grid: all n-tuples of multiples of
    step that are non-negative and sum to 1."""
    reciprocal = round(1.0 / step)
    if step <= 0 or not np.isclose(reciprocal * step, 1.0):
        raise ValueError("simplex step must evenly divide one")
    if n < 1:
        raise ValueError("n must be positive")

    def _recurse(remaining_units: int, slots_left: int) -> list[tuple[int, ...]]:
        if slots_left == 1:
            return [(remaining_units,)]
        results = []
        for units in range(remaining_units + 1):
            for rest in _recurse(remaining_units - units, slots_left - 1):
                results.append((units, *rest))
        return results

    return [
        tuple(units / reciprocal for units in combo)
        for combo in _recurse(reciprocal, n)
    ]


def _seven_station_dir(root: Path) -> Path:
    return Path(root) / "results" / "seven_station_56"


def _parent_manifest_path(root: Path) -> Path:
    return _seven_station_dir(root) / "prediction_manifest.json"


def _verify_parent(root: Path) -> dict:
    manifest = verify_seven_station_prediction_seal(_parent_manifest_path(root))
    for key in (
        "active_target_labels_used_for_training",
        "active_target_labels_used_for_calibration",
        "active_target_labels_used_for_prediction",
    ):
        if manifest.get(key) is not False:
            raise ValueError(f"sealed parent does not declare {key}=false")
    return manifest


def _load_parent_data(root: Path) -> pd.DataFrame:
    root = Path(root).resolve()
    config_path = (
        root / "configs" / "experiments" / "seven_station_56.yaml"
    ).resolve()
    _, data, _ = load_seven_station_data(config_path)
    return data


def _load_parent_config(root: Path) -> dict:
    config_path = (
        Path(root) / "configs" / "experiments" / "seven_station_56.yaml"
    ).resolve()
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _require_isolated_output(root: Path, output_dir: Path) -> None:
    root_results = (Path(root) / "results").resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == root_results or root_results in output_dir.parents:
        forbidden = {"seven_station_56", "seven_station_56_shrinkage"}
        if output_dir.name in forbidden:
            raise ValueError("output_dir must not overwrite a sealed parent")


def _observed_case(data: pd.DataFrame, *, station_id: str, event_id: str) -> pd.DataFrame:
    frame = data.loc[
        (data["station_id"].astype(str) == str(station_id))
        & (data["event_id"].astype(str) == str(event_id)),
        ["timestamp_utc", "water_level_m"],
    ].sort_values("timestamp_utc")
    if len(frame) == 0:
        raise ValueError(f"no observed rows for {station_id}/{event_id}")
    return frame.reset_index(drop=True)


def _range_scale(observed: pd.DataFrame) -> float:
    values = observed["water_level_m"].to_numpy(float)
    scale = float(np.std(values))
    if scale <= 0:
        raise ValueError("observed series has zero variance")
    return scale


def _all_transfer_errors(
    root: Path,
    data: pd.DataFrame,
    *,
    source_pool: list[str],
    calibration_events: list[str],
) -> pd.DataFrame:
    """Every (source, pseudo_target, event) transfer_nrmse within the
    calibration pool, using the frozen seven_station_56 raw predictions.
    Pure source-pool computation -- never touches the active target or test
    event. One row per (source, pseudo_target, event); seeds are averaged
    first so the prior reflects a stable per-pair error, not seed noise."""
    rows = []
    for pseudo_target in source_pool:
        candidates = [s for s in source_pool if s != pseudo_target]
        for event_id in calibration_events:
            observed = _observed_case(data, station_id=pseudo_target, event_id=event_id)
            scale = _range_scale(observed)
            per_seed_errors = {source: [] for source in candidates}
            for seed in SEEDS:
                raw = pd.read_csv(
                    raw_prediction_path(
                        _seven_station_dir(root),
                        target_station_id=pseudo_target,
                        event_id=event_id,
                        seed=int(seed),
                    ),
                    parse_dates=["timestamp_utc"],
                )
                merged = observed.merge(
                    raw[["timestamp_utc", *candidates]],
                    on="timestamp_utc",
                    how="inner",
                    validate="one_to_one",
                )
                if len(merged) != len(observed):
                    raise ValueError("source prior label alignment failed")
                for source in candidates:
                    residual = (
                        merged[source].to_numpy(float)
                        - merged["water_level_m"].to_numpy(float)
                    )
                    per_seed_errors[source].append(
                        float(np.sqrt(np.mean(residual**2))) / scale
                    )
            for source in candidates:
                rows.append(
                    {
                        "source_station_id": source,
                        "pseudo_target_station_id": pseudo_target,
                        "event_id": event_id,
                        "transfer_nrmse": float(np.mean(per_seed_errors[source])),
                    }
                )
    return pd.DataFrame(rows)


def compute_source_priors(
    root: Path,
    data: pd.DataFrame,
    *,
    source_pool: list[str],
    calibration_events: list[str],
) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    """Two views of the same source-history prior, both target-blind:

    - pair_priors[(source, pseudo_target)]: leave-that-pair-out mean
      transfer_nrmse for `source` across every OTHER pseudo-target in the
      pool (excludes the (source, pseudo_target) pair itself, matching
      attach_source_quality_prior's convention -- used during calibration
      so a candidate's prior never encodes its own held-out performance).
    - flat_priors[source]: plain mean transfer_nrmse for `source` across
      the full pool (no pair to exclude, since the real active target is
      never in this pool at all -- used at final prediction time).
    """
    errors = _all_transfer_errors(
        root, data, source_pool=source_pool, calibration_events=calibration_events
    )
    flat_priors: dict[str, float] = {}
    pair_priors: dict[tuple[str, str], float] = {}
    for source in source_pool:
        own = errors.loc[errors["source_station_id"] == source]
        if own.empty:
            raise ValueError(f"no transfer errors computed for source {source}")
        flat_priors[source] = float(own["transfer_nrmse"].mean())
        for pseudo_target in source_pool:
            if pseudo_target == source:
                continue
            remaining = own.loc[own["pseudo_target_station_id"] != pseudo_target]
            if remaining.empty:
                raise ValueError(
                    f"no remaining transfer errors for {source} excluding "
                    f"{pseudo_target}"
                )
            pair_priors[(source, pseudo_target)] = float(
                remaining["transfer_nrmse"].mean()
            )
    return pair_priors, flat_priors


def _standardize_prior(
    priors: dict[str, float], candidate_source_ids: list[str]
) -> dict[str, float]:
    """Standardize source_prior against the candidate pool for one task, so
    it sits on the same roughly-unit scale as the other RMS group distances
    before being combined via group_weights."""
    values = np.array([priors[s] for s in candidate_source_ids], dtype=float)
    mean = float(values.mean())
    scale = float(values.std())
    scale = scale if scale > 1e-9 else 1.0
    return {s: (priors[s] - mean) / scale for s in candidate_source_ids}


def build_group_distances(
    data: pd.DataFrame,
    coastal_full: pd.DataFrame,
    *,
    candidate_source_ids: list[str],
    target_station_id: str,
    event_id: str,
    group_variables: dict[str, list[str]],
    reference: dict,
    coastal_reference_station_ids: list[str],
    stage: StageConfig,
    prior_by_source: dict[str, float] | None,
) -> pd.DataFrame:
    """One row per candidate source, one column per active group in
    stage.group_names(). atmosphere/wave/coastal reuse build_pair_distances
    (unchanged RMS formula); water_fraction/shelf_depth reuse
    compute_static_descriptor_distance; source_prior is standardized
    against this task's candidate pool."""
    base = build_pair_distances(
        data,
        coastal_full,
        candidate_source_ids=candidate_source_ids,
        target_station_id=target_station_id,
        event_id=event_id,
        group_variables=group_variables,
        reference=reference,
    )
    if stage.include_water_fraction:
        base["water_fraction"] = [
            compute_static_descriptor_distance(
                coastal_full,
                source_station_id=source_id,
                target_station_id=target_station_id,
                column=EXTRA_STATIC_COLUMNS["water_fraction_distance"],
                reference_station_ids=coastal_reference_station_ids,
            )
            for source_id in base["source_station_id"]
        ]
    if stage.include_shelf_depth:
        base["shelf_depth"] = [
            compute_static_descriptor_distance(
                coastal_full,
                source_station_id=source_id,
                target_station_id=target_station_id,
                column=EXTRA_STATIC_COLUMNS["shelf_depth_distance"],
                reference_station_ids=coastal_reference_station_ids,
            )
            for source_id in base["source_station_id"]
        ]
    if stage.include_source_prior:
        if prior_by_source is None:
            raise ValueError("stage requires prior_by_source")
        standardized = _standardize_prior(prior_by_source, candidate_source_ids)
        base["source_prior"] = [
            standardized[source_id] for source_id in base["source_station_id"]
        ]
    return base


def compose_forecasts_n(
    predictions: pd.DataFrame,
    information: pd.DataFrame,
    *,
    group_names: tuple[str, ...],
    group_weights: tuple[float, ...],
    temperature: float,
    lambda_value: float,
) -> tuple[dict[str, np.ndarray], dict]:
    """N-group generalization of seven_station_runner.compose_method_forecasts,
    with shrinkage folded directly into the same call (unlike the original
    two-stage hand-crafted pipeline)."""
    source_ids = information["source_station_id"].astype(str).tolist()
    if predictions.columns.tolist() != ["timestamp_utc", *source_ids]:
        raise ValueError("source prediction columns do not match pair information")
    columns = [f"{group}_distance" if group in BASE_GROUP_VARIABLES else group for group in group_names]
    # base groups keep the "_distance" suffix from build_pair_distances;
    # extra groups (water_fraction/shelf_depth/source_prior) are already
    # named without a suffix in build_group_distances.
    resolved_columns = []
    for group in group_names:
        if group in BASE_GROUP_VARIABLES:
            resolved_columns.append(f"{group}_distance")
        else:
            resolved_columns.append(group)
    values = information[resolved_columns].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("similarity distances must be finite")
    combined = values @ np.asarray(group_weights, dtype=float)
    logits = -combined / float(temperature)
    logits -= logits.max()
    scores = np.exp(logits)
    weights = scores / scores.sum()
    equal = np.repeat(1.0 / len(weights), len(weights))
    shrunk = lambda_value * weights + (1.0 - lambda_value) * equal
    shrunk = shrunk / shrunk.sum()

    source_values = predictions[source_ids].to_numpy(float)
    if not np.isfinite(source_values).all():
        raise ValueError("source predictions must be finite")
    nearest_id = str(
        information.loc[information["geo_distance_km"].astype(float).idxmin(), "source_station_id"]
    ) if "geo_distance_km" in information.columns else source_ids[int(np.argmin(combined))]
    top1_id = str(pd.Series(weights, index=source_ids).idxmax())
    methods = {
        "nearest_source": predictions[nearest_id].to_numpy(float),
        "equal_ensemble": source_values.mean(axis=1),
        "similarity_top1": predictions[top1_id].to_numpy(float),
        "similarity_weighted": source_values @ weights,
        "similarity_shrunk": source_values @ shrunk,
    }
    audit = {
        "nearest_source_station_id": nearest_id,
        "top1_source_station_id": top1_id,
        "source_weights": {s: float(w) for s, w in zip(source_ids, weights, strict=True)},
        "shrunk_source_weights": {s: float(w) for s, w in zip(source_ids, shrunk, strict=True)},
    }
    return methods, audit


def _series_nrmse(
    predicted: np.ndarray, observed: np.ndarray, scale: float, *, objective: str
) -> float:
    if objective == "whole_series":
        residual = predicted - observed
        return float(np.sqrt(np.mean(residual**2))) / scale
    if objective == "top10pct":
        threshold = np.quantile(observed, 0.90)
        mask = observed >= threshold
        if mask.sum() < 2:
            mask = np.ones_like(observed, dtype=bool)
        residual = predicted[mask] - observed[mask]
        return float(np.sqrt(np.mean(residual**2))) / scale
    raise ValueError(f"unknown objective: {objective}")


def calibrate_task(
    root: Path,
    data: pd.DataFrame,
    coastal_full: pd.DataFrame,
    *,
    active_target: str,
    test_event: str,
    stage: StageConfig,
) -> dict:
    active_target = str(active_target)
    test_event = str(test_event)
    source_pool = [s for s in STATIONS if s != active_target]
    calibration_events = [e for e in EVENTS if e != test_event]
    coastal_reference_station_ids = sorted(
        str(s) for s in coastal_full["station_id"].unique() if str(s) != active_target
    )
    reference = fit_descriptor_reference(
        data,
        coastal_full,
        source_station_ids=source_pool,
        calibration_events=calibration_events,
        group_variables=BASE_GROUP_VARIABLES,
        coastal_reference_station_ids=coastal_reference_station_ids,
    )

    pair_priors: dict[tuple[str, str], float] = {}
    flat_priors: dict[str, float] = {}
    if stage.include_source_prior:
        pair_priors, flat_priors = compute_source_priors(
            root, data, source_pool=source_pool, calibration_events=calibration_events
        )

    group_names = stage.group_names()
    n_groups = len(group_names)
    grid = simplex_grid_n(stage.grid_step, n_groups)
    weight_grid = np.array(grid, dtype=float)  # (G, n_groups)

    resolved_columns = [
        f"{g}_distance" if g in BASE_GROUP_VARIABLES else g for g in group_names
    ]

    validation_cases = []
    for pseudo_target in source_pool:
        candidates = [s for s in source_pool if s != pseudo_target]
        for event_id in calibration_events:
            observed = _observed_case(data, station_id=pseudo_target, event_id=event_id)
            scale = _range_scale(observed)
            prior_by_source = (
                {c: pair_priors[(c, pseudo_target)] for c in candidates}
                if stage.include_source_prior
                else None
            )
            information = build_group_distances(
                data, coastal_full,
                candidate_source_ids=candidates,
                target_station_id=pseudo_target,
                event_id=event_id,
                group_variables=BASE_GROUP_VARIABLES,
                reference=reference,
                coastal_reference_station_ids=coastal_reference_station_ids,
                stage=stage,
                prior_by_source=prior_by_source,
            )
            ordered = information.set_index("source_station_id").loc[candidates]
            values = ordered[resolved_columns].to_numpy(float)  # (n_candidates, n_groups)
            observed_values = observed["water_level_m"].to_numpy(float)
            for seed in SEEDS:
                raw = pd.read_csv(
                    raw_prediction_path(
                        _seven_station_dir(root), target_station_id=pseudo_target,
                        event_id=event_id, seed=int(seed),
                    ),
                    parse_dates=["timestamp_utc"],
                )
                predictions = raw[["timestamp_utc", *candidates]]
                merged = predictions[["timestamp_utc"]].merge(
                    observed, on="timestamp_utc", how="inner", validate="one_to_one"
                )
                if len(merged) != len(observed):
                    raise ValueError("calibration label alignment failed")
                pred_matrix = raw[candidates].to_numpy(float)  # (216, n_candidates)
                validation_cases.append(
                    {
                        "values": values,
                        "predictions": pred_matrix,
                        "observed": observed_values,
                        "scale": scale,
                    }
                )

    n_cases = len(validation_cases)
    n_grid = len(grid)
    totals = {
        (t_idx, l_idx): np.zeros(n_grid, dtype=float)
        for t_idx in range(len(stage.temperatures))
        for l_idx in range(len(stage.lambdas))
    }
    for case in validation_cases:
        n_candidates = case["values"].shape[0]
        combined_all = case["values"] @ weight_grid.T  # (n_candidates, G)
        equal = 1.0 / n_candidates
        observed_arr = case["observed"]
        if stage.objective == "top10pct":
            threshold = np.quantile(observed_arr, 0.90)
            mask = observed_arr >= threshold
            if mask.sum() < 2:
                mask = np.ones_like(observed_arr, dtype=bool)
        else:
            mask = np.ones_like(observed_arr, dtype=bool)
        observed_masked = observed_arr[mask]
        pred_masked = case["predictions"][mask]
        for t_idx, temperature in enumerate(stage.temperatures):
            logits = -combined_all / float(temperature)
            logits = logits - logits.max(axis=0, keepdims=True)
            scores = np.exp(logits)
            weights = scores / scores.sum(axis=0, keepdims=True)  # (n_candidates, G)
            for l_idx, lambda_value in enumerate(stage.lambdas):
                shrunk = float(lambda_value) * weights + (1.0 - float(lambda_value)) * equal
                shrunk = shrunk / shrunk.sum(axis=0, keepdims=True)
                predicted = pred_masked @ shrunk  # (len(masked), G)
                residual = predicted - observed_masked[:, None]
                nrmse = np.sqrt(np.mean(residual**2, axis=0)) / case["scale"]  # (G,)
                totals[(t_idx, l_idx)] += nrmse

    best = None
    for t_idx, temperature in enumerate(stage.temperatures):
        for l_idx, lambda_value in enumerate(stage.lambdas):
            mean_nrmse = totals[(t_idx, l_idx)] / n_cases  # (G,)
            grid_idx = int(np.argmin(mean_nrmse))
            candidate = (
                float(mean_nrmse[grid_idx]),
                -float(temperature),
                float(lambda_value),
                grid_idx,
                float(temperature),
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    best_nrmse, _, best_lambda, best_grid_idx, best_temperature = best

    return {
        "target_station_id": active_target,
        "test_event": test_event,
        "group_names": list(group_names),
        "group_weights": list(grid[best_grid_idx]),
        "temperature": float(best_temperature),
        "lambda": float(best_lambda),
        "objective": stage.objective,
        "coastal_reference_station_ids": coastal_reference_station_ids,
        "descriptor_reference": reference,
        "source_quality_priors": flat_priors,
        "calibration_station_ids": source_pool,
        "calibration_events": calibration_events,
        "candidate_grid_size": len(grid),
        "validation_case_count": n_cases,
        "selected_mean_nrmse": float(best_nrmse),
        "target_labels_used": False,
        "test_event_labels_used": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_prediction_task(
    root: Path,
    data: pd.DataFrame,
    coastal_full: pd.DataFrame,
    *,
    selection: dict,
    seed: int,
    stage: StageConfig,
) -> tuple[dict, dict]:
    active_target = str(selection["target_station_id"])
    event_id = str(selection["test_event"])
    source_ids = [s for s in STATIONS if s != active_target]
    raw = pd.read_csv(
        raw_prediction_path(_seven_station_dir(root), target_station_id=active_target, event_id=event_id, seed=int(seed)),
        parse_dates=["timestamp_utc"],
    )
    validate_raw_prediction(raw, source_station_ids=source_ids, target_station_id=active_target)
    prior_by_source = (
        {s: selection["source_quality_priors"][s] for s in source_ids}
        if stage.include_source_prior
        else None
    )
    information = build_group_distances(
        data, coastal_full,
        candidate_source_ids=source_ids,
        target_station_id=active_target,
        event_id=event_id,
        group_variables=BASE_GROUP_VARIABLES,
        reference=selection["descriptor_reference"],
        coastal_reference_station_ids=selection["coastal_reference_station_ids"],
        stage=stage,
        prior_by_source=prior_by_source,
    )
    predictions = raw[["timestamp_utc", *source_ids]]
    methods, audit = compose_forecasts_n(
        predictions, information,
        group_names=tuple(selection["group_names"]),
        group_weights=tuple(selection["group_weights"]),
        temperature=float(selection["temperature"]),
        lambda_value=float(selection["lambda"]),
    )
    result = {"timestamp_utc": raw["timestamp_utc"], **methods}
    return result, {
        "target_station_id": active_target,
        "event_id": event_id,
        "seed": int(seed),
        "group_names": selection["group_names"],
        "group_weights": selection["group_weights"],
        "temperature": selection["temperature"],
        "lambda": selection["lambda"],
        **audit,
        "pair_distances": information.to_dict(orient="records"),
        "active_target_labels_used": False,
    }


def _selection_path(output_dir: Path, target: str, event: str) -> Path:
    return output_dir / "calibration" / "selections" / f"target_{target}__event_{event}.json"


def _method_path(output_dir: Path, target: str, event: str, seed: int) -> Path:
    return (
        output_dir / "predictions" / "methods"
        / f"target_{target}__event_{event}__seed_{int(seed)}.csv"
    )


def _audit_path(output_dir: Path, target: str, event: str, seed: int) -> Path:
    return (
        output_dir / "predictions" / "similarity_audit"
        / f"target_{target}__event_{event}__seed_{int(seed)}.json"
    )


def _write_seal(
    manifest_path: Path, *, artifacts: dict[str, list[Path]], metadata: dict
) -> Path:
    manifest_path = Path(manifest_path)
    records: dict[str, list[dict]] = {}
    for role, paths in artifacts.items():
        records[role] = [
            {
                "path": Path(
                    __import__("os").path.relpath(
                        Path(p).resolve(), manifest_path.parent.resolve()
                    )
                ).as_posix(),
                "bytes": Path(p).stat().st_size,
                "sha256": sha256_file(Path(p)),
            }
            for p in paths
        ]
    manifest = {
        **metadata,
        "labels_revealed": False,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def verify_seal(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("labels_revealed") is not False:
        raise ValueError("prediction seal is already revealed or invalid")
    for role, records in manifest["artifacts"].items():
        for record in records:
            path = (manifest_path.parent / record["path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"size mismatch for sealed artifact: {path}")
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"hash mismatch for sealed artifact: {path}")
    return manifest


def run_stage(root: Path, output_dir: Path, stage: StageConfig) -> Path:
    """Calibrate + predict + seal all 56 tasks for one stage. Reuses the
    frozen seven_station_56 raw predictions/checkpoints without retraining."""
    root = Path(root)
    output_dir = Path(output_dir)
    _require_isolated_output(root, output_dir)
    parent_manifest = _verify_parent(root)
    parent_manifest_path = _parent_manifest_path(root)
    data = _load_parent_data(root)
    config = _load_parent_config(root)
    coastal_full = load_full_coastal_context(config)

    selection_paths: list[Path] = []
    method_paths: list[Path] = []
    audit_paths: list[Path] = []
    for index, target in enumerate(STATIONS, start=1):
        for event in EVENTS:
            selection = calibrate_task(
                root, data, coastal_full, active_target=target, test_event=event, stage=stage
            )
            sel_path = _selection_path(output_dir, target, event)
            sel_path.parent.mkdir(parents=True, exist_ok=True)
            sel_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
            selection_paths.append(sel_path)
            for seed in SEEDS:
                result, audit = run_prediction_task(
                    root, data, coastal_full, selection=selection, seed=seed, stage=stage
                )
                method_df = pd.DataFrame(result)[list(METHOD_COLUMNS)]
                m_path = _method_path(output_dir, target, event, seed)
                m_path.parent.mkdir(parents=True, exist_ok=True)
                method_df.to_csv(m_path, index=False)
                method_paths.append(m_path)
                a_path = _audit_path(output_dir, target, event, seed)
                a_path.parent.mkdir(parents=True, exist_ok=True)
                a_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
                audit_paths.append(a_path)
        print(f"[{stage.name}] calibrated+predicted {index}/{len(STATIONS)} stations", flush=True)

    manifest_path = output_dir / "prediction_manifest.json"
    _write_seal(
        manifest_path,
        artifacts={
            "parent_manifest": [parent_manifest_path],
            "selection": selection_paths,
            "method_prediction": method_paths,
            "similarity_audit": audit_paths,
        },
        metadata={
            "stage_name": stage.name,
            "group_names": list(stage.group_names()),
            "grid_step": stage.grid_step,
            "objective": stage.objective,
            "task_count": len(STATIONS) * len(EVENTS),
            "active_target_labels_used_for_calibration": False,
            "active_target_labels_used_for_prediction": False,
        },
    )
    print(f"[{stage.name}] sealed: {manifest_path}", flush=True)
    return manifest_path
