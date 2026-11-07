"""Push Check A's seed-averaging technique to an earlier point in the
pipeline: average the 3 seeds' predicted NON-TIDAL RESIDUAL (NTR) series
BEFORE reconstructing water level (adding back the target's own predicted
tide), instead of averaging the already-reconstructed water-level series
(what Check A and every prior attempt did). Tide reconstruction is a
deterministic, seed-invariant addition, so it should not matter whether it
happens before or after averaging IF the residual and the tide combine
linearly with no other seed-dependent step in between -- and they do
(reconstruct_water_level is a pure elementwise add) -- so this is really a
test of whether averaging strictly in NTR units (which may have a
different noise scale/distribution than reconstructed water level, since
predicted_tide_m itself has a much larger amplitude and could compress the
NTR noise's relative influence in the final scale) surfaces any additional
benefit. Reuses the exact frozen seven_station_56 checkpoints for
inference only (no retraining) -- this does not touch or modify the sealed
parent, it just calls predict_residual_checkpoint again to additionally
capture predicted_ntr_m, which raw_prediction_path's saved CSV never
persisted. Otherwise identical to Check A: same 3 groups, same
grid_step=0.1, same temperature/lambda grids."""

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
from src.experiments.residual_forecasting import (
    build_target_inputs,
    predict_residual_checkpoint,
    reconstruct_water_level,
)
from src.experiments.seven_station_contract import EVENTS, SEEDS, STATIONS
from src.experiments.seven_station_runner import checkpoint_path

STAGE = StageConfig(
    name="handcrafted_seed_averaged_ntr",
    grid_step=0.1,
    temperatures=DEFAULT_TEMPERATURES,
    lambdas=DEFAULT_LAMBDAS,
)


def _ntr_averaged_raw(
    root: Path, data: pd.DataFrame, config: dict,
    *, target_station_id: str, event_id: str, source_ids: list[str],
) -> pd.DataFrame:
    """Average predicted NTR across the 3 seeds, then reconstruct water
    level once from the averaged NTR (inference-only reuse of the frozen
    seven_station_56 checkpoints, no retraining, no seal modification)."""
    features = list(config["features"]["dynamic"])
    target_rows = data.loc[
        (data["station_id"] == target_station_id) & (data["event_id"] == event_id)
    ].sort_values("timestamp_utc")
    target_inputs = build_target_inputs(target_rows, features)
    seven_station_dir = _seven_station_dir(root)

    per_seed_ntr = []
    times_ref = None
    for seed in SEEDS:
        columns = {}
        for source in source_ids:
            job = {"test_event": event_id, "source_station_id": source, "seed": int(seed)}
            path = checkpoint_path(seven_station_dir, job)
            forecast = predict_residual_checkpoint(path, target_inputs)
            if times_ref is None:
                times_ref = forecast["timestamp_utc"].reset_index(drop=True)
            columns[source] = forecast["predicted_ntr_m"].to_numpy(float)
        per_seed_ntr.append(columns)

    averaged_ntr = {
        source: np.mean([seed_cols[source] for seed_cols in per_seed_ntr], axis=0)
        for source in source_ids
    }
    result = {"timestamp_utc": times_ref}
    for source in source_ids:
        result[source] = reconstruct_water_level(
            times_ref, predicted_residual=averaged_ntr[source], target_inputs=target_inputs
        )
    return pd.DataFrame(result)


def calibrate_task(
    root: Path, data: pd.DataFrame, coastal_full: pd.DataFrame, config: dict,
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
            averaged_raw = _ntr_averaged_raw(
                root, data, config, target_station_id=pseudo_target, event_id=event_id, source_ids=candidates
            )
            merged = averaged_raw.merge(observed, on="timestamp_utc", how="inner", validate="one_to_one")
            if len(merged) != len(observed):
                raise ValueError("NTR-averaged calibration label alignment failed")
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


def run_prediction_task(root, data, coastal_full, config, *, selection: dict) -> tuple[pd.DataFrame, dict]:
    active_target = str(selection["target_station_id"])
    event_id = str(selection["test_event"])
    source_ids = [s for s in STATIONS if s != active_target]
    averaged_raw = _ntr_averaged_raw(
        root, data, config, target_station_id=active_target, event_id=event_id, source_ids=source_ids
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
            selection = calibrate_task(root, data, coastal_full, config, active_target=target, test_event=event)
            sel_path = output_dir / "calibration" / "selections" / f"target_{target}__event_{event}.json"
            sel_path.parent.mkdir(parents=True, exist_ok=True)
            sel_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
            selection_paths.append(sel_path)

            result, audit = run_prediction_task(root, data, coastal_full, config, selection=selection)
            for seed in SEEDS:
                m_path = output_dir / "predictions" / "methods" / f"target_{target}__event_{event}__seed_{int(seed)}.csv"
                m_path.parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(m_path, index=False)
                method_paths.append(m_path)
                a_path = output_dir / "predictions" / "similarity_audit" / f"target_{target}__event_{event}__seed_{int(seed)}.json"
                a_path.parent.mkdir(parents=True, exist_ok=True)
                a_path.write_text(json.dumps({**audit, "seed": int(seed)}, ensure_ascii=False, indent=2), encoding="utf-8")
                audit_paths.append(a_path)
        print(f"[handcrafted_seed_averaged_ntr] calibrated+predicted {index}/{len(STATIONS)} stations", flush=True)

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
    print(f"[handcrafted_seed_averaged_ntr] sealed: {manifest_path}", flush=True)
    return manifest_path
