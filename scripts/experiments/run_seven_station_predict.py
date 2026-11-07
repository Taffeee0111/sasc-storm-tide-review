"""Run and seal the complete seven-station by eight-event grid."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.seven_station_contract import (
    SEEDS,
    build_checkpoint_grid,
    build_task_grid,
)
from src.experiments.seven_station_runner import (
    calibrate_task_similarity,
    checkpoint_path,
    load_seven_station_data,
    load_full_coastal_context,
    method_prediction_path,
    raw_prediction_path,
    run_checkpoint_pool,
    run_method_prediction_task,
    run_raw_prediction_task,
    selection_path,
    verify_seven_station_prediction_seal,
    write_seven_station_prediction_seal,
)


def _write_state(path: Path, *, stage: str, completed: int, total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "stage": stage,
                "completed": completed,
                "total": total,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_seven_station_predictions(
    config_path: Path,
    output_dir: Path,
    *,
    max_epochs_override: int | None = None,
) -> Path:
    """Execute all prediction stages with checkpoint/file-level resume."""
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    manifest_path = output_dir / "prediction_manifest.json"
    if manifest_path.is_file():
        verify_seven_station_prediction_seal(manifest_path)
        print(f"Verified existing complete prediction seal: {manifest_path}")
        return manifest_path

    config, data, _ = load_seven_station_data(config_path)
    full_coastal = load_full_coastal_context(config)
    tasks = build_task_grid(config)
    jobs = build_checkpoint_grid(config)
    state_path = output_dir / "run_state.json"

    checkpoint_records = []
    for index, job in enumerate(jobs, start=1):
        record = run_checkpoint_pool(
            config,
            data,
            output_dir,
            jobs=[job],
            max_epochs_override=max_epochs_override,
        )[0]
        checkpoint_records.append(record)
        _write_state(
            state_path,
            stage="checkpoint_pool",
            completed=index,
            total=len(jobs),
        )
        print(
            f"[checkpoint {index:03d}/{len(jobs)}] "
            f"{job['test_event']} {job['source_station_id']} "
            f"seed={job['seed']} {record['status']}",
            flush=True,
        )
    pd.DataFrame(checkpoint_records).to_json(
        output_dir / "checkpoint_run_records.json",
        orient="records",
        indent=2,
    )

    raw_paths = []
    total_raw = len(tasks) * len(SEEDS)
    raw_index = 0
    for task in tasks:
        for seed in SEEDS:
            raw_paths.append(
                run_raw_prediction_task(
                    config,
                    data,
                    output_dir,
                    task=task,
                    seed=seed,
                )
            )
            raw_index += 1
            _write_state(
                state_path,
                stage="raw_transfer_predictions",
                completed=raw_index,
                total=total_raw,
            )
            print(
                f"[raw {raw_index:03d}/{total_raw}] "
                f"{task['target_station_id']} {task['test_event']} seed={seed}",
                flush=True,
            )

    selection_paths = []
    for index, task in enumerate(tasks, start=1):
        print(
            f"[calibration {index:02d}/{len(tasks)}] "
            f"{task['target_station_id']} {task['test_event']} starting",
            flush=True,
        )
        selection_paths.append(
            calibrate_task_similarity(
                config,
                data,
                full_coastal,
                output_dir,
                task=task,
            )
        )
        _write_state(
            state_path,
            stage="task_specific_calibration",
            completed=index,
            total=len(tasks),
        )
        print(
            f"[calibration {index:02d}/{len(tasks)}] completed",
            flush=True,
        )

    method_paths = []
    method_index = 0
    for task, task_selection_path in zip(tasks, selection_paths, strict=True):
        for seed in SEEDS:
            method_paths.append(
                run_method_prediction_task(
                    config,
                    data,
                    full_coastal,
                    output_dir,
                    task=task,
                    seed=seed,
                    selection_path=task_selection_path,
                )
            )
            method_index += 1
            _write_state(
                state_path,
                stage="method_predictions",
                completed=method_index,
                total=total_raw,
            )
            print(
                f"[methods {method_index:03d}/{total_raw}] "
                f"{task['target_station_id']} {task['test_event']} seed={seed}",
                flush=True,
            )

    task_manifest_path = output_dir / "task_manifest.csv"
    pd.DataFrame(tasks).to_csv(task_manifest_path, index=False)
    data_path = ROOT / config["data"]["event_dataset"]
    coastal_path = ROOT / config["data"]["coastal_context"]
    checkpoint_paths = [checkpoint_path(output_dir, job) for job in jobs]
    expected_raw_paths = [
        raw_prediction_path(
            output_dir,
            target_station_id=task["target_station_id"],
            event_id=task["test_event"],
            seed=seed,
        )
        for task in tasks
        for seed in SEEDS
    ]
    expected_method_paths = [
        method_prediction_path(
            output_dir,
            target_station_id=task["target_station_id"],
            event_id=task["test_event"],
            seed=seed,
        )
        for task in tasks
        for seed in SEEDS
    ]
    expected_selection_paths = [selection_path(output_dir, task) for task in tasks]
    audit_paths = sorted(
        (output_dir / "predictions" / "similarity_audit").glob("*.json")
    )
    manifest_path = write_seven_station_prediction_seal(
        manifest_path,
        artifacts={
            "config": [config_path],
            "dataset": [data_path],
            "coastal_context": [coastal_path],
            "checkpoint": checkpoint_paths,
            "raw_prediction": expected_raw_paths,
            "method_prediction": expected_method_paths,
            "calibration_selection": expected_selection_paths,
            "similarity_audit": audit_paths,
        },
        metadata={
            "experiment_id": config["experiment_id"],
            "task_count": len(tasks),
            "seed_count": len(SEEDS),
            "checkpoint_count": len(checkpoint_paths),
            "raw_prediction_count": len(raw_paths),
            "method_prediction_count": len(method_paths),
            "calibration_selection_count": len(selection_paths),
            "active_target_labels_used_for_training": False,
            "active_target_labels_used_for_calibration": False,
            "active_target_labels_used_for_prediction": False,
            "cross_validation_note": (
                "Each station acts as a source in other folds, but the active "
                "target station is excluded from source models and calibration "
                "inside its own target-event fold. The coastal-group "
                "standardization reference spans the full built-station "
                "inventory, not just the six active sources. Two of the seven "
                "stations (Sewells Point, semi_enclosed; Money Point, "
                "enclosed) are deliberately non-open-exposure, to stress-test "
                "whether learned weighting can identify and down-weight "
                "genuinely anomalous candidate sources."
            ),
            "task_manifest_path": str(task_manifest_path),
        },
    )
    verify_seven_station_prediction_seal(manifest_path)
    _write_state(state_path, stage="sealed", completed=1, total=1)
    print(f"Seven-station prediction grid sealed: {manifest_path}", flush=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "seven_station_56.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "seven_station_56",
    )
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()
    run_seven_station_predictions(
        args.config,
        args.output,
        max_epochs_override=args.max_epochs,
    )


if __name__ == "__main__":
    main()
