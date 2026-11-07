"""Seed-averaged variant of the original 3-group hand-crafted calibration:
average the 3 seeds' raw per-source predictions into one series BEFORE
similarity calibration/combination, instead of combining each seed
separately and averaging the resulting RMSE afterward (the original and
every one of the 21 prior attempts this session all did the latter).
Targets backbone training stochasticity (checkpoint-to-checkpoint seed
noise, quantified elsewhere as ~0.0088 NRMSE) as a noise source distinct
from calibration overfitting -- never isolated before. Same 3 groups, same
original grid_step/temperature/lambda grid as the frozen hand-crafted
baseline; only the raw-prediction averaging point changes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.handcrafted_expanded_similarity_runner import (
    BASE_GROUP_VARIABLES,
    DEFAULT_LAMBDAS,
    DEFAULT_TEMPERATURES,
    METHOD_COLUMNS,
    StageConfig,
    _load_parent_config,
    _load_parent_data,
    _observed_case,
    _parent_manifest_path,
    _range_scale,
    _require_isolated_output,
    _seven_station_dir,
    _verify_parent,
    _write_seal,
    build_group_distances,
    compose_forecasts_n,
    load_full_coastal_context,
    simplex_grid_n,
)
from src.experiments.seven_station_contract import EVENTS, SEEDS, STATIONS
from src.experiments.seven_station_runner import raw_prediction_path, validate_raw_prediction


STAGE = StageConfig(
    name="handcrafted_seed_averaged",
    grid_step=0.1,
    temperatures=DEFAULT_TEMPERATURES,
    lambdas=DEFAULT_LAMBDAS,
)


def _seed_averaged_raw(
    root: Path, *, target_station_id: str, event_id: str, source_ids: list[str]
) -> pd.DataFrame:
    """Average the 3 seeds' raw per-source predictions into one series.
    source_ids may be a narrower candidate list than the raw file's native
    columns (all STATIONS except target_station_id) when computing a nested
    calibration fold that additionally excludes a held-out pseudo-target."""
    native_source_ids = [s for s in STATIONS if s != target_station_id]
    frames = []
    for seed in SEEDS:
        raw = pd.read_csv(
            raw_prediction_path(
                _seven_station_dir(root), target_station_id=target_station_id,
                event_id=event_id, seed=int(seed),
            ),
            parse_dates=["timestamp_utc"],
        )
        validate_raw_prediction(raw, source_station_ids=native_source_ids, target_station_id=target_station_id)
        frames.append(raw.set_index("timestamp_utc")[source_ids])
    averaged = sum(frames) / len(frames)
    return averaged.reset_index()


def calibrate_task(
    root: Path, data: pd.DataFrame, coastal_full: pd.DataFrame,
    *, active_target: str, test_event: str,
) -> dict:
    active_target = str(active_target)
    test_event = str(test_event)
    source_pool = [s for s in STATIONS if s != active_target]
    calibration_events = [e for e in EVENTS if e != test_event]
    coastal_reference_station_ids = sorted(
        str(s) for s in coastal_full["station_id"].unique() if str(s) != active_target
    )
    from src.experiments.source_only_calibration import fit_descriptor_reference

    reference = fit_descriptor_reference(
        data, coastal_full, source_station_ids=source_pool,
        calibration_events=calibration_events, group_variables=BASE_GROUP_VARIABLES,
        coastal_reference_station_ids=coastal_reference_station_ids,
    )
    group_names = STAGE.group_names()
    grid = simplex_grid_n(STAGE.grid_step, len(group_names))
    weight_grid = np.array(grid, dtype=float)
    resolved_columns = [f"{g}_distance" for g in group_names]

    validation_cases = []
    for pseudo_target in source_pool:
        candidates = [s for s in source_pool if s != pseudo_target]
        for event_id in calibration_events:
            observed = _observed_case(data, station_id=pseudo_target, event_id=event_id)
            scale = _range_scale(observed)
            information = build_group_distances(
                data, coastal_full, candidate_source_ids=candidates,
                target_station_id=pseudo_target, event_id=event_id,
                group_variables=BASE_GROUP_VARIABLES, reference=reference,
                coastal_reference_station_ids=coastal_reference_station_ids,
                stage=STAGE, prior_by_source=None,
            )
            ordered = information.set_index("source_station_id").loc[candidates]
            values = ordered[resolved_columns].to_numpy(float)
            averaged_raw = _seed_averaged_raw(
                root, target_station_id=pseudo_target, event_id=event_id, source_ids=candidates
            )
            merged = averaged_raw.merge(observed, on="timestamp_utc", how="inner", validate="one_to_one")
            if len(merged) != len(observed):
                raise ValueError("seed-averaged calibration label alignment failed")
            validation_cases.append(
                {
                    "values": values,
                    "predictions": merged[candidates].to_numpy(float),
                    "observed": merged["water_level_m"].to_numpy(float),
                    "scale": scale,
                }
            )

    n_cases = len(validation_cases)
    n_grid = len(grid)
    totals = {
        (t_idx, l_idx): np.zeros(n_grid, dtype=float)
        for t_idx in range(len(STAGE.temperatures)) for l_idx in range(len(STAGE.lambdas))
    }
    for case in validation_cases:
        n_candidates = case["values"].shape[0]
        combined_all = case["values"] @ weight_grid.T
        equal = 1.0 / n_candidates
        observed_arr = case["observed"]
        for t_idx, temperature in enumerate(STAGE.temperatures):
            logits = -combined_all / float(temperature)
            logits = logits - logits.max(axis=0, keepdims=True)
            scores = np.exp(logits)
            weights = scores / scores.sum(axis=0, keepdims=True)
            for l_idx, lambda_value in enumerate(STAGE.lambdas):
                shrunk = float(lambda_value) * weights + (1.0 - float(lambda_value)) * equal
                shrunk = shrunk / shrunk.sum(axis=0, keepdims=True)
                predicted = case["predictions"] @ shrunk
                residual = predicted - observed_arr[:, None]
                nrmse = np.sqrt(np.mean(residual**2, axis=0)) / case["scale"]
                totals[(t_idx, l_idx)] += nrmse

    best = None
    for t_idx, temperature in enumerate(STAGE.temperatures):
        for l_idx, lambda_value in enumerate(STAGE.lambdas):
            mean_nrmse = totals[(t_idx, l_idx)] / n_cases
            grid_idx = int(np.argmin(mean_nrmse))
            candidate = (float(mean_nrmse[grid_idx]), -float(temperature), float(lambda_value), grid_idx, float(temperature))
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    best_nrmse, _, best_lambda, best_grid_idx, best_temperature = best

    return {
        "target_station_id": active_target, "test_event": test_event,
        "group_names": list(group_names), "group_weights": list(grid[best_grid_idx]),
        "temperature": float(best_temperature), "lambda": float(best_lambda),
        "coastal_reference_station_ids": coastal_reference_station_ids,
        "descriptor_reference": reference,
        "selected_mean_nrmse": float(best_nrmse),
        "target_labels_used": False, "test_event_labels_used": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_prediction_task(root, data, coastal_full, *, selection: dict) -> tuple[pd.DataFrame, dict]:
    active_target = str(selection["target_station_id"])
    event_id = str(selection["test_event"])
    source_ids = [s for s in STATIONS if s != active_target]
    averaged_raw = _seed_averaged_raw(
        root, target_station_id=active_target, event_id=event_id, source_ids=source_ids
    )
    information = build_group_distances(
        data, coastal_full, candidate_source_ids=source_ids,
        target_station_id=active_target, event_id=event_id,
        group_variables=BASE_GROUP_VARIABLES, reference=selection["descriptor_reference"],
        coastal_reference_station_ids=selection["coastal_reference_station_ids"],
        stage=STAGE, prior_by_source=None,
    )
    methods, audit = compose_forecasts_n(
        averaged_raw[["timestamp_utc", *source_ids]], information,
        group_names=tuple(selection["group_names"]), group_weights=tuple(selection["group_weights"]),
        temperature=float(selection["temperature"]), lambda_value=float(selection["lambda"]),
    )
    result = pd.DataFrame({"timestamp_utc": averaged_raw["timestamp_utc"], **methods})[list(METHOD_COLUMNS)]
    return result, {
        "target_station_id": active_target, "event_id": event_id,
        "group_names": selection["group_names"], "group_weights": selection["group_weights"],
        "temperature": selection["temperature"], "lambda": selection["lambda"],
        **audit, "active_target_labels_used": False,
    }


def run_all(root: Path, output_dir: Path) -> Path:
    root = Path(root)
    output_dir = Path(output_dir)
    _require_isolated_output(root, output_dir)
    parent_manifest_path = _parent_manifest_path(root)
    _verify_parent(root)
    data = _load_parent_data(root)
    config = _load_parent_config(root)
    coastal_full = load_full_coastal_context(config)

    selection_paths, method_paths, audit_paths = [], [], []
    for index, target in enumerate(STATIONS, start=1):
        for event in EVENTS:
            selection = calibrate_task(root, data, coastal_full, active_target=target, test_event=event)
            sel_path = output_dir / "calibration" / "selections" / f"target_{target}__event_{event}.json"
            sel_path.parent.mkdir(parents=True, exist_ok=True)
            sel_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
            selection_paths.append(sel_path)

            result, audit = run_prediction_task(root, data, coastal_full, selection=selection)
            for seed in SEEDS:
                m_path = output_dir / "predictions" / "methods" / f"target_{target}__event_{event}__seed_{int(seed)}.csv"
                m_path.parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(m_path, index=False)
                method_paths.append(m_path)
                a_path = output_dir / "predictions" / "similarity_audit" / f"target_{target}__event_{event}__seed_{int(seed)}.json"
                a_path.parent.mkdir(parents=True, exist_ok=True)
                a_path.write_text(json.dumps({**audit, "seed": int(seed)}, ensure_ascii=False, indent=2), encoding="utf-8")
                audit_paths.append(a_path)
        print(f"[handcrafted_seed_averaged] calibrated+predicted {index}/{len(STATIONS)} stations", flush=True)

    manifest_path = output_dir / "prediction_manifest.json"
    _write_seal(
        manifest_path,
        artifacts={
            "parent_manifest": [parent_manifest_path],
            "selection": selection_paths, "method_prediction": method_paths, "similarity_audit": audit_paths,
        },
        metadata={
            "stage_name": STAGE.name, "task_count": len(STATIONS) * len(EVENTS),
            "active_target_labels_used_for_calibration": False,
            "active_target_labels_used_for_prediction": False,
        },
    )
    print(f"[handcrafted_seed_averaged] sealed: {manifest_path}", flush=True)
    return manifest_path
