"""Frozen contract for the four-station, eight-event exhaustive experiment."""

from __future__ import annotations

from itertools import product


STATIONS = ["8534720", "8638610", "8670870", "8656483"]
EVENTS = [
    "isabel_2003",
    "irene_2011",
    "sandy_2012",
    "matthew_2016",
    "florence_2018",
    "dorian_2019",
    "isaias_2020",
    "nicole_2022",
]
SEEDS = [20260723, 20260724, 20260725]
EXPECTED_HOURS = 216


def validate_exhaustive_config(config: dict) -> None:
    """Reject any mutation that changes the preregistered experiment grid."""
    if list(config["stations"]["all"]) != STATIONS:
        raise ValueError("exhaustive contract requires the four frozen stations")
    if list(config["events"]["all"]) != EVENTS:
        raise ValueError("exhaustive contract requires the eight frozen events")
    if int(config["events"]["expected_hours_per_event"]) != EXPECTED_HOURS:
        raise ValueError("each station-event window must contain 216 hours")
    if [int(value) for value in config["model"]["random_seeds"]] != SEEDS:
        raise ValueError("exhaustive contract requires the three frozen seeds")
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


def build_event_split(event_id: str, events: list[str] | None = None) -> dict:
    """Use the next frozen event for validation and the other six for training."""
    ordered = list(EVENTS if events is None else events)
    if ordered != EVENTS:
        raise ValueError("event order must match the frozen chronological contract")
    if event_id not in ordered:
        raise ValueError(f"unknown test event: {event_id}")
    index = ordered.index(event_id)
    validation_event = ordered[(index + 1) % len(ordered)]
    train_events = [
        candidate
        for candidate in ordered
        if candidate not in {event_id, validation_event}
    ]
    return {
        "test_event": event_id,
        "validation_event": validation_event,
        "train_events": train_events,
    }


def build_task_grid(config: dict) -> list[dict]:
    """Return all 4×8 target-event tasks with target-blind calibration roles."""
    validate_exhaustive_config(config)
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
    """Return 96 reusable station-event-seed training jobs."""
    validate_exhaustive_config(config)
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
