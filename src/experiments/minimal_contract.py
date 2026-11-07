"""Configuration validation and split construction for the minimal pilot."""

from __future__ import annotations


EXPECTED_SOURCES = ["8534720", "8638610", "8670870"]
EXPECTED_TARGET = "8632200"
EXPECTED_TEST_EVENTS = ["isabel_2003", "sandy_2012"]
EXPECTED_VALIDATION_EVENT = "isaias_2020"


def validate_config(config: dict) -> None:
    """Validate the frozen experiment roles and event policy."""
    sources = [str(value) for value in config["stations"]["sources"]]
    target = str(config["stations"]["target"])
    if sources != EXPECTED_SOURCES:
        raise ValueError(f"source stations must be {EXPECTED_SOURCES}")
    if len(set(sources)) != 3:
        raise ValueError("exactly three unique source stations are required")
    if target in sources:
        raise ValueError("target station must not be a source")
    if target != EXPECTED_TARGET:
        raise ValueError(f"target station must be {EXPECTED_TARGET}")

    tests = list(config["events"]["test"])
    validation = str(config["events"]["validation"])
    if tests != EXPECTED_TEST_EVENTS:
        raise ValueError("test events must be Isabel 2003 and Sandy 2012")
    if validation != EXPECTED_VALIDATION_EVENT:
        raise ValueError("validation event must be Isaias 2020")
    if validation in tests:
        raise ValueError("validation event must not be a test event")

    all_events = list(config["events"]["all"])
    if len(all_events) != len(set(all_events)):
        raise ValueError("event list must not contain duplicates")
    required = set(tests + [validation])
    if not required.issubset(all_events):
        raise ValueError("all events must include validation and test events")


def build_splits(config: dict) -> list[dict]:
    """Return the two frozen leave-event-out splits."""
    validate_config(config)
    all_events = list(config["events"]["all"])
    validation = str(config["events"]["validation"])
    sources = [str(value) for value in config["stations"]["sources"]]
    target = str(config["stations"]["target"])

    splits = []
    for test_event in config["events"]["test"]:
        train_events = [
            event for event in all_events if event not in {test_event, validation}
        ]
        if len(train_events) != 6:
            raise ValueError("each split must contain exactly six training events")
        splits.append(
            {
                "split_id": f"kiptopeke_{test_event}",
                "source_station_ids": sources.copy(),
                "target_station_id": target,
                "train_events": train_events,
                "validation_event": validation,
                "test_event": test_event,
                "target_labels_evaluation_only": True,
            }
        )
    return splits
