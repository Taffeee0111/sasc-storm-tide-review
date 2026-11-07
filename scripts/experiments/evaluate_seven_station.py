"""Reveal labels only after seal verification and evaluate the seven-station
56 tasks (parallel to evaluate_five_station.py; reuses exhaustive_evaluation.py
unchanged since it carries no station constants)."""

from __future__ import annotations

import argparse
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
from src.experiments.exhaustive_evaluation import (
    METHODS,
    aggregate_seed_metrics,
    build_overall_summary,
    build_pairwise_win_rates,
    validate_seed_metric_grid,
)
from src.experiments.seven_station_runner import (
    load_seven_station_data,
    method_prediction_path,
    verify_seven_station_prediction_seal,
)
from src.metrics import evaluate_event


def _evaluate_seed_grid(
    config: dict,
    data: pd.DataFrame,
    prediction_dir: Path,
) -> pd.DataFrame:
    rows: list[dict] = []
    for task in build_task_grid(config):
        target = task["target_station_id"]
        event = task["test_event"]
        observed = data.loc[
            (data["station_id"] == target) & (data["event_id"] == event),
            ["timestamp_utc", "water_level_m"],
        ].sort_values("timestamp_utc")
        for seed in SEEDS:
            path = method_prediction_path(
                prediction_dir,
                target_station_id=target,
                event_id=event,
                seed=seed,
            )
            predictions = pd.read_csv(path, parse_dates=["timestamp_utc"])
            expected_columns = ["timestamp_utc", *METHODS]
            if predictions.columns.tolist() != expected_columns:
                raise ValueError(
                    f"unexpected method prediction columns in {path}"
                )
            paired = observed.merge(
                predictions,
                on="timestamp_utc",
                how="inner",
                validate="one_to_one",
            )
            if len(paired) != 216:
                raise ValueError(f"label alignment did not produce 216 rows: {path}")
            for method in METHODS:
                metrics = evaluate_event(
                    paired["water_level_m"],
                    paired[method],
                    paired["timestamp_utc"],
                )
                rows.append(
                    {
                        "target_station_id": target,
                        "event_id": event,
                        "seed": seed,
                        "method": method,
                        **metrics,
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


def _write_heatmap(task_metrics: pd.DataFrame, path: Path) -> None:
    proposed = task_metrics.loc[
        task_metrics["method"] == "similarity_weighted",
        ["target_station_id", "event_id", "rmse_mean"],
    ].rename(columns={"rmse_mean": "proposed_rmse"})
    equal = task_metrics.loc[
        task_metrics["method"] == "equal_ensemble",
        ["target_station_id", "event_id", "rmse_mean"],
    ].rename(columns={"rmse_mean": "baseline_rmse"})
    merged = proposed.merge(
        equal, on=["target_station_id", "event_id"], validate="one_to_one"
    )
    merged["relative_rmse_gain_pct"] = (
        100.0
        * (merged["baseline_rmse"] - merged["proposed_rmse"])
        / merged["baseline_rmse"]
    )
    matrix = merged.pivot(
        index="target_station_id",
        columns="event_id",
        values="relative_rmse_gain_pct",
    ).reindex(index=STATIONS, columns=EVENTS)
    figure, axis = plt.subplots(figsize=(13, 6.4))
    image = axis.imshow(matrix.to_numpy(float), cmap="RdYlGn", aspect="auto")
    axis.set_xticks(range(len(EVENTS)), [event.replace("_", "\n") for event in EVENTS])
    axis.set_yticks(range(len(STATIONS)), STATIONS)
    axis.set_xlabel("Held-out storm event")
    axis.set_ylabel("Target station")
    axis.set_title("Similarity-weighted RMSE gain over equal ensemble (%)")
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


def _write_method_boxplot(task_metrics: pd.DataFrame, path: Path) -> None:
    values = [
        task_metrics.loc[task_metrics["method"] == method, "rmse_mean"].to_numpy(float)
        for method in METHODS
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.boxplot(values, tick_labels=METHODS, showmeans=True)
    axis.set_ylabel("Task-mean RMSE (m)")
    axis.set_title("Seven-station closed-book RMSE distribution across 56 target-event tasks")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: (
                ""
                if pd.isna(value)
                else f"{float(value):.{decimals}f}"
            )
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


def _write_summary(
    path: Path,
    overall: pd.DataFrame,
    wins: pd.DataFrame,
) -> None:
    best = overall.sort_values("rmse_task_mean").iloc[0]
    weighted = overall.loc[
        overall["method"] == "similarity_weighted"
    ].iloc[0]
    lines = [
        "# Seven-station 56-task closed-book evaluation",
        "",
        "## Protocol",
        "",
        "- 7 target stations: 5 open-coast stations (Atlantic City, "
        "Kiptopeke, Beaufort, Fort Pulaski, Trident Pier) plus two "
        "deliberately non-open-exposure stations -- Sewells Point "
        "(semi_enclosed) and Money Point (enclosed) -- x 8 held-out storm "
        "events x 3 fixed seeds.",
        "- This is a stress test: every task's candidate source pool now "
        "includes at least one genuinely hydrologically anomalous station "
        "for some fraction of targets. Equal-weight ensembling must include "
        "every candidate at 1/6 weight regardless; learned weighting can, "
        "in principle, down-weight or exclude an anomalous source.",
        "- Chesapeake Bay Bridge Tunnel remains excluded because its event "
        "data is incomplete (4 of 8 storms missing).",
        "- Active target/event labels were revealed only after prediction seal verification.",
        "- Every task compares the same six source forecasts under four source-selection/fusion rules.",
        "- All 56 tasks are retained; failures are not removed.",
        "",
        "## Headline result",
        "",
        f"- Lowest mean task RMSE: **{best['method']}** "
        f"({best['rmse_task_mean']:.4f} m).",
        f"- Similarity-weighted mean task RMSE: "
        f"**{weighted['rmse_task_mean']:.4f} m**.",
        "",
        "## Similarity-weighted pairwise win rates",
        "",
        _markdown_table(wins, decimals=3),
        "",
        "## Overall task-level summary",
        "",
        _markdown_table(overall, decimals=4),
        "",
        "Interpretation must use both mean performance and task-level win rate. "
        "A few favorable storms are not sufficient evidence of a general improvement.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_seven_station_predictions(
    prediction_dir: Path,
    config_path: Path,
) -> Path:
    """Verify the sealed predictions first, then reveal and evaluate labels."""
    prediction_dir = Path(prediction_dir)
    seal_path = prediction_dir / "prediction_manifest.json"
    seal = verify_seven_station_prediction_seal(seal_path)

    # Label-bearing data is loaded only after the complete prediction seal passed.
    config, data, _ = load_seven_station_data(config_path)
    seed_metrics = _evaluate_seed_grid(config, data, prediction_dir)
    task_metrics = aggregate_seed_metrics(seed_metrics)
    overall = build_overall_summary(task_metrics)
    wins = build_pairwise_win_rates(task_metrics)

    metrics_dir = prediction_dir / "metrics"
    figures_dir = prediction_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    seed_path = metrics_dir / "seed_level_metrics.csv"
    task_path = metrics_dir / "task_method_mean_std.csv"
    overall_path = metrics_dir / "overall_method_summary.csv"
    wins_path = metrics_dir / "weighted_pairwise_win_rates.csv"
    appendix_path = metrics_dir / "full_appendix.csv"
    seed_metrics.to_csv(seed_path, index=False)
    task_metrics.to_csv(task_path, index=False)
    overall.to_csv(overall_path, index=False)
    wins.to_csv(wins_path, index=False)
    task_metrics.sort_values(
        ["target_station_id", "event_id", "method"]
    ).to_csv(appendix_path, index=False)
    _write_heatmap(task_metrics, figures_dir / "weighted_gain_heatmap.png")
    _write_method_boxplot(task_metrics, figures_dir / "method_rmse_boxplot.png")
    summary_path = prediction_dir / "seven_station_summary.md"
    _write_summary(summary_path, overall, wins)

    manifest_path = prediction_dir / "evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "prediction_seal_verified_before_label_reveal": True,
                "prediction_seal_created_utc": seal["sealed_utc"],
                "labels_revealed_for_evaluation": True,
                "seed_metric_rows": int(len(seed_metrics)),
                "task_method_rows": int(len(task_metrics)),
                "task_count": 56,
                "method_count": len(METHODS),
                "output_files": [
                    str(path.relative_to(prediction_dir))
                    for path in (
                        seed_path,
                        task_path,
                        overall_path,
                        wins_path,
                        appendix_path,
                        figures_dir / "weighted_gain_heatmap.png",
                        figures_dir / "method_rmse_boxplot.png",
                        summary_path,
                    )
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Evaluation complete: {summary_path}", flush=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "results" / "seven_station_56",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "seven_station_56.yaml",
    )
    args = parser.parse_args()
    evaluate_seven_station_predictions(args.predictions, args.config)


if __name__ == "__main__":
    main()
