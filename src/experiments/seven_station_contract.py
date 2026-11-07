"""Frozen contract for the seven-station, eight-event heterogeneous-exposure
experiment (parallel to five_station_contract.py, adding Sewells Point, VA
(8638610, semi_enclosed) and Money Point, VA (8639348, enclosed) -- two
non-open-exposure stations with complete 8-event data that were unused by
the five-station open-coast panel. This is a deliberate stress test: the
panel now contains candidate sources that are genuinely hydrologically
different from most targets, not just more open-coast stations."""

from __future__ import annotations

from itertools import product

from src.experiments.exhaustive_contract import (
    EVENTS,
    EXPECTED_HOURS,
    SEEDS,
    build_event_split,
)

STATIONS = [
    "8534720",
    "8632200",
    "8656483",
    "8670870",
    "8721604",
    "8638610",
    "8639348",
]


def validate_seven_station_config(config: dict) -> None:
    """Reject any mutation that changes the preregistered seven-station grid."""
    if list(config["stations"]["all"]) != STATIONS:
        raise ValueError("seven-station contract requires the seven frozen stations")
    if list(config["events"]["all"]) != EVENTS:
        raise ValueError("seven-station contract requires the eight frozen events")
    if int(config["events"]["expected_hours_per_event"]) != EXPECTED_HOURS:
        raise ValueError("each station-event window must contain 216 hours")
    if [int(value) for value in config["model"]["random_seeds"]] != SEEDS:
        raise ValueError("seven-station contract requires the three frozen seeds")
    if str(config["model"]["label_column"]) != "ntr_m":
        raise ValueError("source models must predict non-tidal residual ntr_m")
    if config["model"]["reconstruct_water_level_with_predicted_tide"] is not True:
        raise ValueError("water level must be reconstructed with target predicted tide")
    if config["similarity"]["include_geographic_distance"] is not False:
        raise ValueError("geographic distance is baseline-only")
    if config["data_policy"]["allow_new_downloads"] is not False:
        raise ValueError("new downloads are forbidden")
    if config["data_policy"]["target_labels_evaluation_only"] is not True:
        raise ValueError("active target labels must remain evaluation-only")
    temperatures = [float(value) for value in config["similarity"]["temperatures"]]
    if not temperatures or any(value <= 0 for value in temperatures):
        raise ValueError("similarity temperatures must be positive")
    step = float(config["similarity"]["grid_step"])
    if step <= 0 or abs(round(1.0 / step) * step - 1.0) > 1e-9:
        raise ValueError("similarity grid step must evenly divide one")


def build_task_grid(config: dict) -> list[dict]:
    """Return all 7x8 target-event tasks with target-blind calibration roles."""
    validate_seven_station_config(config)
    tasks = []
    for target_station_id, test_event in product(STATIONS, EVENTS):
        split = build_event_split(test_event)
        source_station_ids = [
            station_id for station_id in STATIONS if station_id != target_station_id
        ]
        tasks.append(
            {
                "task_id": f"target_{target_station_id}__event_{test_event}",
                "target_station_id": target_station_id,
                "source_station_ids": source_station_ids,
                "test_event": test_event,
                "validation_event": split["validation_event"],
                "train_events": split["train_events"],
                "calibration_station_ids": source_station_ids.copy(),
                "calibration_events": split["train_events"].copy(),
                "target_labels_evaluation_only": True,
            }
        )
    return tasks


def build_checkpoint_grid(config: dict) -> list[dict]:
    """Return 168 reusable station-event-seed training jobs."""
    validate_seven_station_config(config)
    jobs = []
    for test_event, source_station_id, seed in product(EVENTS, STATIONS, SEEDS):
        split = build_event_split(test_event)
        jobs.append(
            {
                "test_event": test_event,
                "source_station_id": source_station_id,
                "seed": seed,
                "validation_event": split["validation_event"],
                "train_events": split["train_events"],
            }
        )
    return jobs
