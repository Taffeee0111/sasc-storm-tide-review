"""Verify seven-station shrinkage seals, reveal labels, and evaluate the
exact 56 tasks (parallel to evaluate_five_station_shrinkage.py; reuses
shrinkage_evaluation.py unchanged since it carries no station constants)."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.seven_station_contract import EVENTS, SEEDS, STATIONS, build_task_grid
from src.experiments.seven_station_runner import (
    load_seven_station_data,
    verify_seven_station_prediction_seal,
)
from src.experiments.shrinkage_evaluation import (
    METHODS,
    aggregate_seed_metrics,
    build_overall_summary,
    build_pairwise_win_rates,
    evaluate_success_gate,
    validate_seed_metric_grid,
)
from src.experiments.seven_station_shrinkage_runner import (
    shrinkage_method_prediction_path,
    shrinkage_selection_path,
    verify_shrinkage_prediction_seal,
)
from src.metrics import evaluate_event


def _markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{decimals}f}"
        )
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _evaluate(
    config: dict,
    data: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for task in build_task_grid(config):
        target = task["target_station_id"]
        event = task["test_event"]
        observed = data.loc[
            (data["station_id"] == target) & (data["event_id"] == event),
            ["timestamp_utc", "water_level_m"],
        ].sort_values("timestamp_utc")
        for seed in SEEDS:
            path = shrinkage_method_prediction_path(
                output_dir,
                target_station_id=target,
                event_id=event,
                seed=seed,
            )
            predictions = pd.read_csv(path, parse_dates=["timestamp_utc"])
            if predictions.columns.tolist() != ["timestamp_utc", *METHODS]:
                raise ValueError(f"unexpected shrinkage method columns: {path}")
            paired = observed.merge(
                predictions,
                on="timestamp_utc",
                how="inner",
                validate="one_to_one",
            )
            if len(paired) != 216:
                raise ValueError(f"shrinkage label alignment failed: {path}")
            for method in METHODS:
                rows.append(
                    {
                        "target_station_id": target,
                        "event_id": event,
                        "seed": seed,
                        "method": method,
                        **evaluate_event(
                            paired["water_level_m"],
                            paired[method],
                            paired["timestamp_utc"],
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    validate_seed_metric_grid(
        result,
        target_ids=STATIONS,
        event_ids=EVENTS,
        seeds=SEEDS,
    )
    return result


def _write_lambda_figure(selections: list[dict], path: Path) -> None:
    counts = Counter(float(selection["lambda"]) for selection in selections)
    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(
        [str(value) for value in values],
        [counts[value] for value in values],
        color="#3a7ca5",
    )
    axis.bar_label(bars)
    axis.set_xlabel("Selected shrinkage lambda")
    axis.set_ylabel("Number of target-event tasks")
    axis.set_title("Source-only calibrated shrinkage intensity (56 seven-station tasks)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_gain_heatmap(task_metrics: pd.DataFrame, path: Path) -> None:
    proposed = task_metrics.loc[
        task_metrics["method"] == "similarity_shrunk",
        ["target_station_id", "event_id", "rmse_mean"],
    ].rename(columns={"rmse_mean": "proposed"})
    equal = task_metrics.loc[
        task_metrics["method"] == "equal_ensemble",
        ["target_station_id", "event_id", "rmse_mean"],
    ].rename(columns={"rmse_mean": "equal"})
    merged = proposed.merge(
        equal,
        on=["target_station_id", "event_id"],
        validate="one_to_one",
    )
    merged["gain_pct"] = 100.0 * (
        merged["equal"] - merged["proposed"]
    ) / merged["equal"]
    matrix = merged.pivot(
        index="target_station_id",
        columns="event_id",
        values="gain_pct",
    ).reindex(index=STATIONS, columns=EVENTS)
    figure, axis = plt.subplots(figsize=(13, 6.4))
    image = axis.imshow(matrix.to_numpy(float), cmap="RdYlGn", aspect="auto")
    axis.set_xticks(
        range(len(EVENTS)), [event.replace("_", "\n") for event in EVENTS]
    )
    axis.set_yticks(range(len(STATIONS)), STATIONS)
    axis.set_xlabel("Held-out storm event")
    axis.set_ylabel("Target station")
    axis.set_title("Seven-station shrinkage RMSE gain over equal ensemble (%)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix.iloc[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="RMSE gain (%)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def evaluate_seven_station_shrinkage_experiment(
    config_path: Path,
    parent_dir: Path,
    output_dir: Path,
) -> Path:
    parent_dir = Path(parent_dir)
    output_dir = Path(output_dir)
    parent_seal = verify_seven_station_prediction_seal(
        parent_dir / "prediction_manifest.json"
    )
    shrinkage_seal = verify_shrinkage_prediction_seal(
        output_dir / "prediction_manifest.json"
    )

    # Label-bearing data is loaded only after both prediction seals verify.
    config, data, _ = load_seven_station_data(config_path)
    seed_metrics = _evaluate(config, data, output_dir)
    task_metrics = aggregate_seed_metrics(seed_metrics)
    overall = build_overall_summary(task_metrics)
    wins = build_pairwise_win_rates(task_metrics)
    shrink_mean = float(
        overall.loc[
            overall["method"] == "similarity_shrunk", "rmse_task_mean"
        ].iloc[0]
    )
    equal_win = float(
        wins.loc[
            wins["baseline_method"] == "equal_ensemble", "rmse_win_rate"
        ].iloc[0]
    )
    gate = evaluate_success_gate(shrink_mean, equal_win)

    tasks = build_task_grid(config)
    selections = [
        json.loads(
            shrinkage_selection_path(output_dir, task).read_text(encoding="utf-8")
        )
        for task in tasks
    ]
    lambda_counts = Counter(float(item["lambda"]) for item in selections)

    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    seed_path = metrics_dir / "seed_level_metrics.csv"
    task_path = metrics_dir / "task_method_mean_std.csv"
    overall_path = metrics_dir / "overall_method_summary.csv"
    wins_path = metrics_dir / "pairwise_win_rates.csv"
    seed_metrics.to_csv(seed_path, index=False)
    task_metrics.to_csv(task_path, index=False)
    overall.to_csv(overall_path, index=False)
    wins.to_csv(wins_path, index=False)
    lambda_path = figures_dir / "lambda_frequency.png"
    heatmap_path = figures_dir / "shrinkage_gain_heatmap.png"
    _write_lambda_figure(selections, lambda_path)
    _write_gain_heatmap(task_metrics, heatmap_path)

    summary_path = output_dir / "shrinkage_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Seven-station equal-shrinkage closed-book evaluation",
                "",
                "## Preregistered gate",
                "",
                f"- Mean task RMSE: **{gate['mean_rmse_m']:.4f} m** "
                f"(must be < {gate['rmse_threshold_m']:.4f} m): "
                f"**{'PASS' if gate['rmse_passed'] else 'FAIL'}**.",
                f"- RMSE win rate versus equal ensemble: "
                f"**{gate['equal_rmse_win_rate']:.3f}** "
                f"(must be > {gate['win_rate_threshold']:.2f}): "
                f"**{'PASS' if gate['win_rate_passed'] else 'FAIL'}**.",
                f"- Overall preregistered result: "
                f"**{'PASS' if gate['passed'] else 'FAIL'}**.",
                "",
                "## Lambda frequency",
                "",
                ", ".join(
                    f"`{value:.2f}`: {lambda_counts[value]}"
                    for value in (0.0, 0.25, 0.5, 0.75, 1.0)
                ),
                "",
                "## Pairwise win rates",
                "",
                _markdown_table(wins, decimals=3),
                "",
                "## Overall task-level summary",
                "",
                _markdown_table(overall, decimals=4),
                "",
                "All 56 tasks are retained. Active target labels were read "
                "only after both prediction seals passed.",
            ]
        ),
        encoding="utf-8",
    )
    audit_path = output_dir / "independent_audit.md"
    audit_path.write_text(
        "\n".join(
            [
                "# Seven-station equal-shrinkage independent audit",
                "",
                f"- Parent prediction seal verified: yes "
                f"({parent_seal['method_prediction_count']} method files).",
                f"- Shrinkage prediction seal verified: yes "
                f"({shrinkage_seal['method_prediction_count']} method files).",
                "- Selections: 56.",
                "- Shrinkage predictions: 168.",
                "- Shrinkage audits: 168.",
                f"- Seed-level metric rows: {len(seed_metrics)}.",
                f"- Task-method metric rows: {len(task_metrics)}.",
                "- Duplicate seed keys: "
                f"{seed_metrics.duplicated(['target_station_id', 'event_id', 'seed', 'method']).sum()}.",
                "- Active target/test labels used in calibration: no.",
                f"- Preregistered gate passed: {'yes' if gate['passed'] else 'no'}.",
                "",
                "No task, station, storm, or seed was excluded after evaluation.",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "parent_seal_verified_before_label_reveal": True,
                "shrinkage_seal_verified_before_label_reveal": True,
                "labels_revealed_for_evaluation": True,
                "seed_metric_rows": len(seed_metrics),
                "task_method_rows": len(task_metrics),
                "success_gate": gate,
                "lambda_counts": {
                    str(value): lambda_counts[value]
                    for value in (0.0, 0.25, 0.5, 0.75, 1.0)
                },
                "output_files": [
                    str(path.relative_to(output_dir))
                    for path in (
                        seed_path,
                        task_path,
                        overall_path,
                        wins_path,
                        lambda_path,
                        heatmap_path,
                        summary_path,
                        audit_path,
                    )
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Seven-station shrinkage evaluation complete: {summary_path}", flush=True)
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
    evaluate_seven_station_shrinkage_experiment(args.config, args.parent, args.output)


if __name__ == "__main__":
    main()
