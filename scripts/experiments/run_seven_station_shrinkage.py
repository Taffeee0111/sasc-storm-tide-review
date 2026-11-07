"""Calibrate, predict, and seal all seven-station source-only shrinkage tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.seven_station_contract import SEEDS, build_task_grid
from src.experiments.seven_station_runner import (
    load_seven_station_data,
    sha256_file,
    verify_seven_station_prediction_seal,
)
from src.experiments.seven_station_shrinkage_runner import (
    calibrate_task_shrinkage,
    run_shrunk_prediction_task,
    shrinkage_audit_path,
    shrinkage_method_prediction_path,
    verify_shrinkage_prediction_seal,
    write_shrinkage_prediction_seal,
)


def run_shrinkage_experiment(
    config_path: Path,
    parent_dir: Path,
    output_dir: Path,
) -> Path:
    parent_dir = Path(parent_dir)
    output_dir = Path(output_dir)
    manifest_path = output_dir / "prediction_manifest.json"
    if manifest_path.is_file():
        verify_shrinkage_prediction_seal(manifest_path)
        print(f"Verified existing shrinkage seal: {manifest_path}", flush=True)
        return manifest_path

    parent_manifest_path = parent_dir / "prediction_manifest.json"
    verify_seven_station_prediction_seal(parent_manifest_path)
    parent_hash_before = sha256_file(parent_manifest_path)
    config, data, _ = load_seven_station_data(config_path)
    tasks = build_task_grid(config)

    selection_paths = []
    for index, task in enumerate(tasks, start=1):
        path = calibrate_task_shrinkage(
            config,
            data,
            parent_dir,
            output_dir,
            task=task,
        )
        selection_paths.append(path)
        print(
            f"[shrinkage calibration {index:02d}/{len(tasks)}] "
            f"{task['target_station_id']} {task['test_event']}",
            flush=True,
        )

    prediction_paths = []
    prediction_index = 0
    for task, selection_path in zip(tasks, selection_paths, strict=True):
        for seed in SEEDS:
            prediction_paths.append(
                run_shrunk_prediction_task(
                    parent_dir,
                    output_dir,
                    task=task,
                    seed=seed,
                    selection_path=selection_path,
                )
            )
            prediction_index += 1
            print(
                f"[shrinkage prediction {prediction_index:03d}/"
                f"{len(tasks) * len(SEEDS)}] "
                f"{task['target_station_id']} {task['test_event']} seed={seed}",
                flush=True,
            )

    audit_paths = [
        shrinkage_audit_path(
            output_dir,
            target_station_id=task["target_station_id"],
            event_id=task["test_event"],
            seed=seed,
        )
        for task in tasks
        for seed in SEEDS
    ]
    expected_prediction_paths = [
        shrinkage_method_prediction_path(
            output_dir,
            target_station_id=task["target_station_id"],
            event_id=task["test_event"],
            seed=seed,
        )
        for task in tasks
        for seed in SEEDS
    ]
    parent_hash_after = sha256_file(parent_manifest_path)
    if parent_hash_after != parent_hash_before:
        raise ValueError("parent prediction manifest changed during shrinkage run")

    write_shrinkage_prediction_seal(
        manifest_path,
        artifacts={
            "parent_manifest": [parent_manifest_path],
            "selection": selection_paths,
            "method_prediction": expected_prediction_paths,
            "shrinkage_audit": audit_paths,
        },
        metadata={
            "experiment_id": "seven_station_56_equal_shrinkage",
            "parent_experiment_id": config["experiment_id"],
            "parent_manifest_sha256": parent_hash_before,
            "task_count": len(tasks),
            "seed_count": len(SEEDS),
            "selection_count": len(selection_paths),
            "method_prediction_count": len(prediction_paths),
            "active_target_labels_used_for_calibration": False,
            "active_target_labels_used_for_prediction": False,
        },
    )
    verify_shrinkage_prediction_seal(manifest_path)
    verify_seven_station_prediction_seal(parent_manifest_path)
    print(f"Shrinkage prediction grid sealed: {manifest_path}", flush=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "seven_station_56.yaml",
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=ROOT / "results" / "seven_station_56",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "seven_station_56_shrinkage",
    )
    args = parser.parse_args()
    run_shrinkage_experiment(args.config, args.parent, args.output)


if __name__ == "__main__":
    main()
