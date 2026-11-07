"""Target-blind Stage-2 finalization router.

This module is the canonical production path for producing the final
per-task storm-tide water-level predictions.  It implements the closed-book
protocol stated in the manuscript: during the prediction phase the active
target station's records are never read, in any form.

Protocol enforced here
----------------------
For an active task ``(q, e)`` (target station ``q``, test event ``e``) the
forbidden label set is ::

    F(q, e) = {(s, v) : s == q  or  v == e}

i.e. the target station's observed water level for *any* event, and any
station's observed water level for the test event.  All label reads during
the prediction phase go through :class:`PseudoTargetLabelStore`, which
rejects any key in ``F(q, e)`` with :class:`TargetLabelLeakError`.

Everything else in the prediction phase is a pure function of quantities
that were frozen before the target's record is consulted:

* ``base``      -- seed-averaged Stage-1 ``similarity_shrunk`` blend
                   (label-free hydrometeorological similarity weighting);
* ``graph``     -- the frozen graph-branch model output, trained with
                   pseudo-target masking (the active station's edge is
                   removed in its own forward pass);
* ``graph_hull``-- the graph output clipped into the anchor hull;
* ``anchor``    -- the timestep-wise median of the seed-averaged raw
                   candidate forecasts.

Router diagnostics are computed from these quantities alone (base-vs-graph
disagreement), and all router parameters are selected per active target by
leave-one-pseudo-target-out validation on the six non-target stations'
own records (the complement of ``F(q, e)``).  The active target's labels
are read only by :func:`run_evaluate`, after every prediction has
been written to disk and sealed with a SHA-256 manifest.

"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Repo layout (portable: resolved relative to this file, no machine-specific
# absolute paths).
# --------------------------------------------------------------------------

MODULE_PATH = Path(__file__).resolve()
PROJECT_ROOT = MODULE_PATH.parent.parent

STAGE1_METHODS_DIR = PROJECT_ROOT / "results" / "handcrafted_seed_averaged" / "predictions" / "methods"
STAGE1_NTR_DIR = PROJECT_ROOT / "results" / "handcrafted_seed_averaged_ntr" / "predictions" / "methods"
GRAPH_INPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "backbone_followups"
    / "stgcn_lite_017i_phase_direction_bonus_b0p2_c0p5_r0p5_f0p05"
    / "predictions"
)
TRUTH_PATH = PROJECT_ROOT / "data" / "processed" / "preexperiment" / "event_dataset.csv"

FILE_PATTERN = re.compile(r"target_(?P<station>\d+)__event_(?P<event>.+?)__seed_(?P<seed>\d+)\.csv")
STAGE1_SEEDS = (20260723, 20260724, 20260725)

PROTOCOL_VERSION = "target-blind-router-1.0"


# --------------------------------------------------------------------------
# Frozen router specification.
#
# Every constant in this block is fixed *a priori* by the protocol.  None of
# them is tuned on, or selected by, any active target's own record.  The
# only per-task quantities selected during prediction are the router
# parameters in PARAM_GRID below, and they are chosen by
# leave-one-pseudo-target-out validation on the six non-target stations.
# --------------------------------------------------------------------------

# Whitelist (surgical bounded peak relocation).
WHITELIST_MIN_ABS_DELTA_TIMING_H = 24.0   # base and graph peaks disagree by >= 24 h
WHITELIST_MAX_TASKS = 2                   # paper: at most three; frozen at two
WHITELIST_MAX_LEVEL_DROP_M = 0.24         # graph peak at most this far below base peak
WHITELIST_ANCHOR_GAP_M = 0.15             # when anchor peak == base peak, anchor-graph peak gap
WHITELIST_SURGERY = {
    "suppress_strength": 0.6,
    "support_strength": 1.2,
    "origin_radius": 1,
    "target_radius": 3,
    "center_bonus": 0.2,
}

# Graph-timing route (move weight from the base peak toward the graph peak).
GRAPH_TIMING_SOFT_CUTOFF_H = 3.0          # below this, peaks are treated as coincident
GRAPH_TIMING_ORIGIN_RADIUS = 1
GRAPH_TIMING_TARGET_RADIUS = 3

# Height route (lift the base toward the anchor near its own peak).
HEIGHT_MIN_LEVEL_GAIN_M = 0.02            # graph peak at least this much above base peak
HEIGHT_MIN_SUPPORT = 0.30                 # label-free graph support near the base peak
HEIGHT_PEAK_RADIUS = 3

# Leave-one-pseudo-target-out grid.  Same role as the grid searched by the
# legacy pipeline, but validated exclusively on non-target stations.
PARAM_GRID = {
    "graph_strength": [0.5, 0.75, 1.0],
    "clipped_mix": [0.5, 0.75, 1.0],
    "target_support": [0.3, 0.6],
    "height_strength": [0.3, 0.5, 0.7],
    "graph_timing_cutoff_h": [4.0, 6.0, 8.0],
}

# Detection threshold used by the post-seal evaluator.  This is the same
# global 1.274 m constant the legacy evaluator used, kept for continuity of
# the POD/FAR/CSI reporting.
DEFAULT_PEAK_THRESHOLD_M = 1.274


# --------------------------------------------------------------------------
# Guarded label access
# --------------------------------------------------------------------------

class TargetLabelLeakError(RuntimeError):
    """Raised when prediction-phase code tries to read a forbidden label."""


@dataclass(frozen=True)
class TaskKey:
    station: str
    event: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.station, self.event)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.station}::{self.event}"


class PseudoTargetLabelStore:
    """Label access for the prediction phase.

    Only pseudo-target keys ``(p, f)`` with ``p != q`` and ``f != e`` may be
    read; any other key raises :class:`TargetLabelLeakError`.  The same
    object is reused for every active target with a per-call forbidden set.
    """

    def __init__(self, truth_frame: pd.DataFrame):
        self._frame = truth_frame

    def labels(self, station: str, event: str, timestamps: pd.Series, active: TaskKey) -> np.ndarray:
        if station == active.station or event == active.event:
            raise TargetLabelLeakError(
                f"forbidden label read for ({station}, {event}) while active target is "
                f"({active.station}, {active.event}); F(q,e) excludes s==q or v==e"
            )
        rows = self._frame.loc[
            (self._frame["station_id"] == station) & (self._frame["event_id"] == event),
            ["timestamp_utc", "water_level_m"],
        ].sort_values("timestamp_utc")
        merged = pd.DataFrame({"timestamp_utc": pd.to_datetime(timestamps, utc=True)}).merge(
            rows, on="timestamp_utc", how="left"
        )
        values = merged["water_level_m"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"pseudo-target labels for ({station}, {event}) are missing or non-finite")
        return values


def load_truth_frame(path: Path | None = None) -> pd.DataFrame:
    """Load the single source of observed water levels.

    Used by the pseudo-target label store during prediction and by the
    evaluator after sealing.  This function itself reads labels; callers are
    responsible for using it in the correct phase.
    """
    frame = pd.read_csv(
        path or TRUTH_PATH,
        dtype={"station_id": str, "event_id": str},
        parse_dates=["timestamp_utc"],
    )
    return frame


# --------------------------------------------------------------------------
# Frozen input loaders (label-free)
# --------------------------------------------------------------------------

def discover_tasks() -> list[TaskKey]:
    keys: set[tuple[str, str]] = set()
    for seed in STAGE1_SEEDS:
        for path in sorted(STAGE1_METHODS_DIR.glob("*.csv")):
            match = FILE_PATTERN.match(path.name)
            if not match:
                continue
            if int(match.group("seed")) == seed:
                keys.add((str(match.group("station")), str(match.group("event"))))
    return [TaskKey(station=s, event=e) for s, e in sorted(keys)]


def load_stage1_base(task: TaskKey) -> tuple[pd.Series, np.ndarray]:
    """Seed-averaged Stage-1 ``similarity_shrunk`` water-level blend."""
    series: list[np.ndarray] = []
    timestamps: pd.Series | None = None
    for seed in STAGE1_SEEDS:
        path = STAGE1_METHODS_DIR / f"target_{task.station}__event_{task.event}__seed_{seed}.csv"
        frame = pd.read_csv(path)
        if timestamps is None:
            timestamps = frame["timestamp_utc"].reset_index(drop=True)
        elif not (frame["timestamp_utc"].reset_index(drop=True).astype(str) == timestamps.astype(str)).all():
            raise ValueError(f"stage-1 seed files disagree on timestamps for {task}")
        series.append(frame["similarity_shrunk"].to_numpy(dtype=float))
    assert timestamps is not None
    stacked = np.stack(series)
    return timestamps, stacked.mean(axis=0)


def load_stage1_ntr(task: TaskKey) -> np.ndarray | None:
    """Seed-averaged Stage-1 ``similarity_shrunk`` non-tidal residual blend."""
    if not STAGE1_NTR_DIR.is_dir():
        return None
    series: list[np.ndarray] = []
    for seed in STAGE1_SEEDS:
        path = STAGE1_NTR_DIR / f"target_{task.station}__event_{task.event}__seed_{seed}.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        series.append(frame["similarity_shrunk"].to_numpy(dtype=float))
    return np.stack(series).mean(axis=0)


def load_graph_inputs(task: TaskKey) -> tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """Frozen graph-branch output, its anchor-hull clip, and the anchor consensus."""
    matches = sorted(GRAPH_INPUT_DIR.glob(f"target_{task.station}__event_{task.event}__seed_*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one graph input file for {task.station}::{task.event}, found {len(matches)}"
        )
    frame = pd.read_csv(matches[0])
    timestamps = frame["timestamp_utc"].reset_index(drop=True)
    graph = frame["graph_pred_water_level_m"].to_numpy(dtype=float)
    graph_hull = frame["graph_clipped_to_hull_water_level_m"].to_numpy(dtype=float)
    anchor = frame["anchor_consensus_water_level_m"].to_numpy(dtype=float)
    return timestamps, graph, graph_hull, anchor


@dataclass(frozen=True)
class TaskDiagnostics:
    """Per-task label-free diagnostics (the router's only inputs)."""

    key: TaskKey
    timestamps: tuple
    base: tuple
    graph: tuple
    graph_hull: tuple
    anchor: tuple
    base_ntr: tuple | None
    base_peak_idx: int
    graph_peak_idx: int
    anchor_peak_idx: int
    delta_timing_h: float
    delta_height_m: float
    graph_support_peak_mean: float
    anchor_peak_gap_m: float

    @property
    def n(self) -> int:
        return len(self.base)


def _as_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def compute_diagnostics(
    key: TaskKey,
    timestamps: pd.Series,
    base: np.ndarray,
    graph: np.ndarray,
    graph_hull: np.ndarray,
    anchor: np.ndarray,
    base_ntr: np.ndarray | None = None,
) -> TaskDiagnostics:
    base = _as_array(base)
    graph = _as_array(graph)
    graph_hull = _as_array(graph_hull)
    anchor = _as_array(anchor)
    if not (len(base) == len(graph) == len(graph_hull) == len(anchor)):
        raise ValueError("input arrays must have equal length")

    b_idx = int(np.argmax(base))
    g_idx = int(np.argmax(graph))
    a_idx = int(np.argmax(anchor))

    delta_t = float(g_idx - b_idx)
    delta_h = float(graph[g_idx] - base[b_idx])

    window = np.arange(len(base))
    peak_mask = (window >= b_idx - 12) & (window <= b_idx + 12)
    support = np.clip(np.maximum(graph - base, 0.0) / np.maximum(anchor - base, 1e-9), 0.0, 1.0)
    support_peak_mean = float(np.mean(support[peak_mask])) if np.any(peak_mask) else 0.0

    anchor_gap = abs(float(anchor[a_idx] - graph[g_idx])) if a_idx == b_idx else 0.0

    return TaskDiagnostics(
        key=key,
        timestamps=tuple(timestamps.tolist()),
        base=tuple(base.tolist()),
        graph=tuple(graph.tolist()),
        graph_hull=tuple(graph_hull.tolist()),
        anchor=tuple(anchor.tolist()),
        base_ntr=tuple(base_ntr.tolist()) if base_ntr is not None else None,
        base_peak_idx=b_idx,
        graph_peak_idx=g_idx,
        anchor_peak_idx=a_idx,
        delta_timing_h=delta_t,
        delta_height_m=delta_h,
        graph_support_peak_mean=support_peak_mean,
        anchor_peak_gap_m=anchor_gap,
    )


def build_diagnostics(tasks: Iterable[TaskKey]) -> dict[tuple[str, str], TaskDiagnostics]:
    cache: dict[tuple[str, str], TaskDiagnostics] = {}
    for task in tasks:
        timestamps, base = load_stage1_base(task)
        ntr = load_stage1_ntr(task)
        graph_timestamps, graph, graph_hull, anchor = load_graph_inputs(task)
        if len(timestamps) != len(base):
            raise ValueError(f"timestamp/prediction length mismatch for {task}")
        if not (graph_timestamps.astype(str) == timestamps.astype(str)).all():
            raise ValueError(f"stage-1 and graph input timestamps disagree for {task}")
        cache[task.key] = compute_diagnostics(task, timestamps, base, graph, graph_hull, anchor, ntr)
    return cache


# --------------------------------------------------------------------------
# Router: subgroup assignment (label-free)
# --------------------------------------------------------------------------

def triangle_window(length: int, center: int, radius: int) -> np.ndarray:
    if radius <= 0:
        radius = 1
    idx = np.arange(length, dtype=float)
    weight = 1.0 - np.abs(idx - float(center)) / float(radius)
    return np.clip(weight, 0.0, 1.0)


@dataclass(frozen=True)
class RouterParams:
    graph_strength: float = 0.75
    clipped_mix: float = 0.75
    target_support: float = 0.6
    height_strength: float = 0.5
    graph_timing_cutoff_h: float = 6.0

    def to_dict(self) -> dict[str, float]:
        return {
            "graph_strength": self.graph_strength,
            "clipped_mix": self.clipped_mix,
            "target_support": self.target_support,
            "height_strength": self.height_strength,
            "graph_timing_cutoff_h": self.graph_timing_cutoff_h,
        }


def _whitelist_qualifies(diag: TaskDiagnostics) -> tuple[bool, float]:
    """Label-free whitelist eligibility plus a fixed diagnostic score."""
    if abs(diag.delta_timing_h) < WHITELIST_MIN_ABS_DELTA_TIMING_H:
        return False, 0.0
    if diag.delta_height_m < -WHITELIST_MAX_LEVEL_DROP_M:
        return False, 0.0
    if diag.anchor_peak_gap_m > WHITELIST_ANCHOR_GAP_M:
        return False, 0.0
    score = abs(diag.delta_timing_h) + 0.5 * max(-diag.delta_height_m, 0.0)
    return True, float(score)


def build_whitelist(diags: dict[tuple[str, str], TaskDiagnostics]) -> set[tuple[str, str]]:
    """Top-``WHITELIST_MAX_TASKS`` qualifying tasks by the fixed diagnostic score.

    Pure function of label-free diagnostics; no label is consulted.
    """
    qualifying: list[tuple[float, tuple[str, str]]] = []
    for key, diag in diags.items():
        ok, score = _whitelist_qualifies(diag)
        if ok:
            qualifying.append((score, key))
    qualifying.sort(key=lambda item: (-item[0], item[1]))
    return {key for _, key in qualifying[:WHITELIST_MAX_TASKS]}


def assign_subgroup(diag: TaskDiagnostics, params: RouterParams, whitelist: set[tuple[str, str]]) -> str:
    if diag.key.key in whitelist:
        return "whitelist"
    delta_t = abs(diag.delta_timing_h)
    if delta_t >= GRAPH_TIMING_SOFT_CUTOFF_H:
        if delta_t >= params.graph_timing_cutoff_h:
            return "graph_timing_strong"
        return "graph_timing_soft"
    if diag.delta_height_m > HEIGHT_MIN_LEVEL_GAIN_M and diag.graph_support_peak_mean >= HEIGHT_MIN_SUPPORT:
        return "height"
    return "neutral"


# --------------------------------------------------------------------------
# Router: route transforms (bounded, label-free)
# --------------------------------------------------------------------------

def _hull_clip(values: np.ndarray, base: np.ndarray, graph: np.ndarray, graph_hull: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    upper = np.maximum.reduce([base, graph, graph_hull, anchor])
    lower = np.minimum.reduce([base, graph, graph_hull, anchor])
    return np.minimum(np.maximum(values, lower), upper)


def whitelist_route(diag: TaskDiagnostics) -> tuple[np.ndarray, float]:
    """Bounded peak relocation from the base peak toward the graph peak."""
    base = _as_array(diag.base)
    graph = _as_array(diag.graph)
    anchor = _as_array(diag.anchor)
    graph_hull = _as_array(diag.graph_hull)
    spec = WHITELIST_SURGERY

    origin_weight = triangle_window(diag.n, diag.base_peak_idx, spec["origin_radius"])
    target_weight = triangle_window(diag.n, diag.graph_peak_idx, spec["target_radius"])
    support_target = np.maximum(graph, base)

    adjusted = base.copy()
    origin_excess = origin_weight * np.maximum(adjusted - graph, 0.0)
    adjusted -= spec["suppress_strength"] * origin_excess

    target_need = target_weight * np.maximum(support_target - adjusted, 0.0)
    adjusted += spec["support_strength"] * target_need

    adjusted[diag.graph_peak_idx] += spec["center_bonus"] * max(
        graph[diag.graph_peak_idx] - adjusted[diag.graph_peak_idx], 0.0
    )

    adjusted = _hull_clip(adjusted, base, graph, graph_hull, anchor)
    trigger = float(np.mean(origin_weight) + np.mean(target_weight))
    return adjusted, trigger


def graph_timing_route(diag: TaskDiagnostics, params: RouterParams) -> tuple[np.ndarray, float]:
    """Blend the base toward the (possibly clipped) graph output near both peaks."""
    base = _as_array(diag.base)
    graph = _as_array(diag.graph)
    graph_hull = _as_array(diag.graph_hull)
    anchor = _as_array(diag.anchor)

    graph_target = (1.0 - params.clipped_mix) * graph + params.clipped_mix * graph_hull
    origin_weight = triangle_window(diag.n, diag.base_peak_idx, GRAPH_TIMING_ORIGIN_RADIUS)
    target_weight = triangle_window(diag.n, diag.graph_peak_idx, GRAPH_TIMING_TARGET_RADIUS)

    adjusted = base + params.graph_strength * origin_weight * (graph_target - base)
    adjusted += params.target_support * target_weight * np.maximum(graph - base, 0.0)

    adjusted = _hull_clip(adjusted, base, graph, graph_hull, anchor)
    trigger = float(np.mean(origin_weight) + np.mean(target_weight))
    return adjusted, trigger


def height_route(diag: TaskDiagnostics, params: RouterParams) -> tuple[np.ndarray, float]:
    """Lift the base toward the anchor near its own peak, bounded by both."""
    base = _as_array(diag.base)
    anchor = _as_array(diag.anchor)

    gate = triangle_window(diag.n, diag.base_peak_idx, HEIGHT_PEAK_RADIUS)
    adjusted = base + params.height_strength * gate * np.maximum(anchor - base, 0.0)

    upper = np.maximum(base, anchor)
    lower = np.minimum(base, anchor)
    adjusted = np.minimum(np.maximum(adjusted, lower), upper)
    trigger = float(np.mean(gate))
    return adjusted, trigger


def apply_router(
    diag: TaskDiagnostics,
    params: RouterParams,
    whitelist: set[tuple[str, str]],
) -> tuple[np.ndarray, str, float]:
    subgroup = assign_subgroup(diag, params, whitelist)
    if subgroup == "whitelist":
        adjusted, trigger = whitelist_route(diag)
    elif subgroup in {"graph_timing_strong", "graph_timing_soft"}:
        adjusted, trigger = graph_timing_route(diag, params)
    elif subgroup == "height":
        adjusted, trigger = height_route(diag, params)
    else:
        adjusted = _as_array(diag.base).copy()
        trigger = 0.0
    return adjusted, subgroup, trigger


# --------------------------------------------------------------------------
# Leave-one-pseudo-target-out parameter selection
# --------------------------------------------------------------------------

def iter_param_grid(grid: dict[str, Sequence[float]] | None = None) -> list[RouterParams]:
    grid = grid or PARAM_GRID
    combos: list[RouterParams] = []
    for gs in grid["graph_strength"]:
        for cm in grid["clipped_mix"]:
            for ts in grid["target_support"]:
                for hs in grid["height_strength"]:
                    for cutoff in grid["graph_timing_cutoff_h"]:
                        combos.append(
                            RouterParams(
                                graph_strength=float(gs),
                                clipped_mix=float(cm),
                                target_support=float(ts),
                                height_strength=float(hs),
                                graph_timing_cutoff_h=float(cutoff),
                            )
                        )
    return combos


def select_params_for_task(
    diags: dict[tuple[str, str], TaskDiagnostics],
    active: TaskKey,
    label_store: PseudoTargetLabelStore,
    whitelist: set[tuple[str, str]],
    grid: dict[str, Sequence[float]] | None = None,
) -> tuple[RouterParams, float]:
    """Select router parameters for ``active`` on its 42 pseudo-target tasks.

    Labels are read exclusively through :class:`PseudoTargetLabelStore`,
    which rejects every key in ``F(active)``.
    """
    pseudo_keys = [
        key
        for key, diag in diags.items()
        if key[0] != active.station and key[1] != active.event
    ]
    if len(pseudo_keys) != 42:
        raise ValueError(f"expected 42 pseudo-target tasks for {active}, found {len(pseudo_keys)}")

    # Label reads and array conversions happen once per pseudo-task, not once
    # per grid point: the labels do not depend on the router parameters.
    pseudo_cache: list[tuple[TaskDiagnostics, np.ndarray, float]] = []
    for key in pseudo_keys:
        diag = diags[key]
        labels = label_store.labels(key[0], key[1], pd.Series(diag.timestamps), active)
        scale = float(np.std(labels))
        pseudo_cache.append((diag, labels, scale))

    candidates = iter_param_grid(grid)
    best: tuple[RouterParams, float] | None = None
    for params in candidates:
        nrmse_values: list[float] = []
        for diag, labels, scale in pseudo_cache:
            adjusted, _, _ = apply_router(diag, params, whitelist)
            rmse = float(np.sqrt(np.mean((adjusted - labels) ** 2)))
            nrmse = rmse / scale if scale > 0.0 else float("inf")
            nrmse_values.append(nrmse)
        mean_nrmse = float(np.mean(nrmse_values))
        if best is None or mean_nrmse < best[1] - 1e-12:
            best = (params, mean_nrmse)
    assert best is not None
    return best


# --------------------------------------------------------------------------
# Finalization and sealing
# --------------------------------------------------------------------------

def finalize_task_predictions(
    diags: dict[tuple[str, str], TaskDiagnostics],
    active: TaskKey,
    label_store: PseudoTargetLabelStore,
    whitelist: set[tuple[str, str]],
    grid: dict[str, Sequence[float]] | None = None,
) -> tuple[pd.DataFrame, RouterParams, float, dict]:
    diag = diags[active.key]
    params, validation_nrmse = select_params_for_task(diags, active, label_store, whitelist, grid)
    adjusted, subgroup, trigger = apply_router(diag, params, whitelist)

    if diag.base_ntr is not None:
        tide = _as_array(diag.base) - _as_array(diag.base_ntr)
        predicted_ntr = adjusted - tide
    else:
        predicted_ntr = np.full_like(adjusted, np.nan)

    frame = pd.DataFrame(
        {
            "timestamp_utc": list(diag.timestamps),
            "predicted_water_level_m": adjusted,
            "predicted_ntr_m": predicted_ntr,
            "base_predicted_water_level_m": _as_array(diag.base),
            "graph_pred_water_level_m": _as_array(diag.graph),
            "graph_clipped_to_hull_water_level_m": _as_array(diag.graph_hull),
            "anchor_consensus_water_level_m": _as_array(diag.anchor),
            "router_subgroup": subgroup,
            "router_trigger_mean": trigger,
            "whitelist_hit": int(active.key in whitelist),
            "delta_timing_h": diag.delta_timing_h,
            "delta_height_m": diag.delta_height_m,
        }
    )
    record = {
        "task": str(active),
        "subgroup": subgroup,
        "whitelist_hit": bool(active.key in whitelist),
        "params": params.to_dict(),
        "validation_mean_nrmse": validation_nrmse,
        "delta_timing_h": diag.delta_timing_h,
        "delta_height_m": diag.delta_height_m,
        "graph_support_peak_mean": diag.graph_support_peak_mean,
    }
    return frame, params, validation_nrmse, record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal_predictions(output_dir: Path, records: list[dict]) -> Path:
    """Write the seal manifest over the saved prediction files."""
    prediction_dir = output_dir / "predictions"
    files = {
        path.name: sha256_file(path)
        for path in sorted(prediction_dir.glob("*.csv"))
    }
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task_count": len(files),
        "files": files,
        "router_records": records,
    }
    seal_path = output_dir / "predictions_seal.json"
    seal_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return seal_path


def run_finalize(
    output_dir: Path,
    tasks: Sequence[TaskKey] | None = None,
    grid: dict[str, Sequence[float]] | None = None,
    truth_provider: Callable[[], pd.DataFrame] | None = None,
) -> tuple[Path, dict]:
    """Prediction phase: finalize every task, then seal.  Never reads the
    active target's record (see :class:`PseudoTargetLabelStore`)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)

    tasks = list(tasks) if tasks is not None else discover_tasks()
    diags = build_diagnostics(tasks)
    whitelist = build_whitelist(diags)

    truth_frame = (truth_provider or load_truth_frame)()
    label_store = PseudoTargetLabelStore(truth_frame)

    records: list[dict] = []
    for active in tasks:
        frame, params, validation_nrmse, record = finalize_task_predictions(
            diags, active, label_store, whitelist, grid
        )
        frame.to_csv(
            output_dir / "predictions" / f"target_{active.station}__event_{active.event}__seed_20260723.csv",
            index=False,
        )
        records.append(record)

    seal_path = seal_predictions(output_dir, records)
    return seal_path, {"task_count": len(tasks), "whitelist": sorted(whitelist)}


# --------------------------------------------------------------------------
# Post-seal evaluation (the only place the active target's record may be read)
# --------------------------------------------------------------------------

def _event_metrics(observed: np.ndarray, predicted: np.ndarray, timestamps: pd.Series, peak_threshold: float) -> dict[str, float]:
    from src.metrics import evaluate_event  # pure label math; imported lazily to keep the module import-light

    metrics = evaluate_event(observed, predicted, timestamps)
    observed_exceeds = observed >= peak_threshold
    predicted_exceeds = predicted >= peak_threshold
    hits = int(np.sum(observed_exceeds & predicted_exceeds))
    misses = int(np.sum(observed_exceeds & (~predicted_exceeds)))
    false_alarms = int(np.sum((~observed_exceeds) & predicted_exceeds))
    pod = float("nan") if hits + misses == 0 else float(hits / (hits + misses))
    far = float("nan") if hits + false_alarms == 0 else float(false_alarms / (hits + false_alarms))
    csi = float("nan") if hits + misses + false_alarms == 0 else float(hits / (hits + misses + false_alarms))
    return {
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "kge": float(metrics["kge"]),
        "nse": float(metrics["nse"]),
        "pbias": float(metrics["pbias"]),
        "rsr": float(metrics["rsr"]),
        "willmott_d": float(metrics["willmott_d"]),
        "peak_level_error_m": float(metrics["peak_level_error_m"]),
        "peak_time_error_h": float(metrics["peak_time_error_h"]),
        "pod": pod,
        "far": far,
        "csi": csi,
    }


def verify_seal(output_dir: Path) -> dict:
    seal_path = output_dir / "predictions_seal.json"
    if not seal_path.exists():
        raise FileNotFoundError(f"seal manifest missing in {output_dir}; run finalize first")
    manifest = json.loads(seal_path.read_text(encoding="utf-8"))
    prediction_dir = output_dir / "predictions"
    for name, expected in manifest["files"].items():
        actual = sha256_file(prediction_dir / name)
        if actual != expected:
            raise ValueError(f"seal check failed for {name}: prediction file changed after sealing")
    return manifest


def run_evaluate(
    output_dir: Path,
    peak_threshold: float | None = None,
    truth_provider: Callable[[], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Evaluation phase.  Reads the seal, then -- and only then -- the target
    labels, and computes per-task plus aggregate metrics."""
    manifest = verify_seal(output_dir)
    truth_frame = (truth_provider or load_truth_frame)()

    rows: list[dict] = []
    for name in sorted((output_dir / "predictions").glob("*.csv")):
        match = FILE_PATTERN.match(name.name)
        if not match:
            continue
        station, event = str(match.group("station")), str(match.group("event"))
        frame = pd.read_csv(name, parse_dates=["timestamp_utc"])
        truth = truth_frame.loc[
            (truth_frame["station_id"] == station) & (truth_frame["event_id"] == event),
            ["timestamp_utc", "water_level_m"],
        ].sort_values("timestamp_utc")
        merged = frame.merge(truth, on="timestamp_utc", how="inner")
        if len(merged) != len(frame):
            raise ValueError(f"truth alignment failed for {station}::{event}")
        metrics = _event_metrics(
            merged["water_level_m"].to_numpy(float),
            merged["predicted_water_level_m"].to_numpy(float),
            merged["timestamp_utc"],
            peak_threshold if peak_threshold is not None else DEFAULT_PEAK_THRESHOLD_M,
        )
        rows.append({"task": f"{station}::{event}", **metrics})

    per_task = pd.DataFrame(rows)
    summary = {
        "task_count": int(len(per_task)),
        "mean_task_rmse": float(per_task["rmse"].mean()),
        "mean_task_mae": float(per_task["mae"].mean()),
        "mean_task_kge": float(per_task["kge"].mean()),
        "mean_task_nse": float(per_task["nse"].mean()),
        "mean_task_pbias": float(per_task["pbias"].mean()),
        "mean_task_rsr": float(per_task["rsr"].mean()),
        "mean_task_willmott_d": float(per_task["willmott_d"].mean()),
        "mean_peak_level_error_m": float(per_task["peak_level_error_m"].mean()),
        "mean_peak_time_error_h": float(per_task["peak_time_error_h"].mean()),
        "mean_abs_peak_level_error_m": float(per_task["peak_level_error_m"].abs().mean()),
        "mean_abs_peak_time_error_h": float(per_task["peak_time_error_h"].abs().mean()),
        "mean_pod": float(per_task["pod"].mean()),
        "mean_far": float(per_task["far"].mean()),
        "mean_csi": float(per_task["csi"].mean()),
        "peak_threshold_m": float(peak_threshold if peak_threshold is not None else DEFAULT_PEAK_THRESHOLD_M),
        "seal_protocol_version": manifest.get("protocol_version"),
    }
    (output_dir / "metrics").mkdir(exist_ok=True)
    per_task.to_csv(output_dir / "metrics" / "per_task_metrics.csv", index=False)
    (output_dir / "metrics" / "overall_summary.csv").write_text(
        pd.DataFrame([summary]).to_csv(index=False), encoding="utf-8"
    )
    return per_task, summary
