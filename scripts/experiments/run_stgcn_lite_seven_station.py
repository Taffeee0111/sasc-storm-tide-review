from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.seven_station_contract import STATIONS, EVENTS, build_task_grid
from src.metrics import evaluate_event
from src.models.stgcn_lite import STGCNLite

COASTAL_COLUMNS = [
    "coastline_orientation_sin2",
    "coastline_orientation_cos2",
    "nearshore_log_depth",
    "shelf_slope_m_per_km",
    "open_ray_fraction_50km",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class GlobalScalers:
    feature_scaler: StandardScaler
    label_scaler: StandardScaler
    static_scaler: StandardScaler


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * radius * np.arcsin(np.sqrt(a)))


def fit_scalers(frame: pd.DataFrame, feature_columns: list[str]) -> GlobalScalers:
    feature_scaler = StandardScaler().fit(frame[feature_columns].to_numpy(float))
    label_scaler = StandardScaler().fit(frame[["ntr_m"]].to_numpy(float))
    static_scaler = StandardScaler().fit(frame[["station_id", *COASTAL_COLUMNS]].drop_duplicates("station_id")[COASTAL_COLUMNS].to_numpy(float))
    return GlobalScalers(feature_scaler=feature_scaler, label_scaler=label_scaler, static_scaler=static_scaler)


def load_data(config_path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = pd.read_csv(
        ROOT / config["data"]["event_dataset"],
        dtype={"station_id": str, "event_id": str},
        parse_dates=["timestamp_utc"],
    )
    coastal = pd.read_csv(
        ROOT / config["data"]["coastal_context"],
        dtype={"station_id": str},
    )
    data = data.loc[data["station_id"].isin(STATIONS) & data["event_id"].isin(EVENTS)].copy()
    coastal = coastal.loc[coastal["station_id"].isin(STATIONS)].copy()
    return config, data, coastal


def event_matrix(
    data: pd.DataFrame,
    *,
    station_ids: list[str],
    event_id: str,
    feature_columns: list[str],
    scalers: GlobalScalers,
    include_labels: bool,
) -> tuple[np.ndarray, np.ndarray | None, pd.DatetimeIndex]:
    feature_series = []
    label_series = []
    reference_times = None
    for station_id in station_ids:
        frame = data.loc[(data["station_id"] == station_id) & (data["event_id"] == event_id)].sort_values("timestamp_utc").copy()
        if len(frame) != 216:
            raise ValueError(f"{station_id}/{event_id} requires 216 rows")
        times = pd.DatetimeIndex(frame["timestamp_utc"])
        if reference_times is None:
            reference_times = times
        elif not reference_times.equals(times):
            raise ValueError(f"timestamps misaligned for {event_id}")
        feature_values = scalers.feature_scaler.transform(frame[feature_columns].to_numpy(float)).astype(np.float32)
        feature_series.append(feature_values)
        if include_labels:
            label_values = scalers.label_scaler.transform(frame[["ntr_m"]].to_numpy(float)).ravel().astype(np.float32)
            label_series.append(label_values)
    x = np.stack(feature_series, axis=0)
    y = None if not include_labels else np.stack(label_series, axis=0)
    return x, y, reference_times


def build_graph_windows(
    data: pd.DataFrame,
    *,
    station_ids: list[str],
    event_ids: list[str],
    feature_columns: list[str],
    lookback: int,
    scalers: GlobalScalers,
    include_labels: bool,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    windows = []
    labels = []
    metadata_rows = []
    for event_id in event_ids:
        event_x, event_y, times = event_matrix(
            data,
            station_ids=station_ids,
            event_id=event_id,
            feature_columns=feature_columns,
            scalers=scalers,
            include_labels=include_labels,
        )
        for index in range(event_x.shape[1]):
            start = max(0, index - lookback + 1)
            window = event_x[:, start : index + 1, :]
            if window.shape[1] < lookback:
                pad = np.repeat(window[:, [0], :], repeats=lookback - window.shape[1], axis=1)
                window = np.concatenate([pad, window], axis=1)
            windows.append(window.astype(np.float32))
            if include_labels and event_y is not None:
                labels.append(event_y[:, index].astype(np.float32))
            metadata_rows.append({"event_id": event_id, "timestamp_utc": times[index]})
    x = np.stack(windows).astype(np.float32)
    y = None if not include_labels else np.stack(labels).astype(np.float32)
    return x, y, pd.DataFrame(metadata_rows)


def build_static_inputs(
    coastal: pd.DataFrame,
    data: pd.DataFrame,
    *,
    station_ids: list[str],
    scalers: GlobalScalers,
) -> tuple[np.ndarray, np.ndarray]:
    coastal_map = coastal.set_index("station_id")
    coords = data[["station_id", "latitude", "longitude"]].drop_duplicates("station_id").set_index("station_id")
    static = coastal_map.loc[station_ids, COASTAL_COLUMNS].to_numpy(float)
    static_scaled = scalers.static_scaler.transform(static).astype(np.float32)
    geo = np.zeros((len(station_ids), len(station_ids)), dtype=np.float32)
    for i, left in enumerate(station_ids):
        for j, right in enumerate(station_ids):
            geo[i, j] = haversine_km(
                float(coords.loc[left, "latitude"]),
                float(coords.loc[left, "longitude"]),
                float(coords.loc[right, "latitude"]),
                float(coords.loc[right, "longitude"]),
            )
    positive_geo = geo[geo > 0]
    geo_scale = float(np.median(positive_geo)) if len(positive_geo) else 1.0
    geo_kernel = np.exp(-geo / max(geo_scale, 1.0)).astype(np.float32)
    coastal_dist = np.linalg.norm(static_scaled[:, None, :] - static_scaled[None, :, :], axis=-1)
    positive_coastal = coastal_dist[coastal_dist > 0]
    coastal_scale = float(np.median(positive_coastal)) if len(positive_coastal) else 1.0
    coastal_kernel = np.exp(-(coastal_dist ** 2) / max(coastal_scale ** 2, 1e-6)).astype(np.float32)
    adj = 0.5 * geo_kernel + 0.5 * coastal_kernel
    np.fill_diagonal(adj, 1.0)
    adj = adj / np.clip(adj.sum(axis=1, keepdims=True), 1e-6, None)
    return static_scaled, adj.astype(np.float32)


def loader(x: np.ndarray, y: np.ndarray, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(0).expand_as(pred)
    return torch.nn.functional.smooth_l1_loss(pred[expanded], target[expanded])


def validation_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, static_tensor: torch.Tensor, adj_tensor: torch.Tensor, batch_size: int, mask: torch.Tensor) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for inputs, labels in loader(x, y, batch_size=batch_size, shuffle=False, seed=0):
            preds = model(inputs.float(), static_tensor, adj_tensor)
            loss = masked_huber(preds, labels.float(), mask)
            total += float(loss.item()) * len(labels)
            count += len(labels)
    return total / max(count, 1)


def train_task_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    static_tensor: torch.Tensor,
    adj_tensor: torch.Tensor,
    dynamic_in_dim: int,
    static_in_dim: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[STGCNLite, list[dict[str, float]]]:
    set_seed(seed)
    model = STGCNLite(dynamic_in_dim=dynamic_in_dim, static_in_dim=static_in_dim, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    mask = torch.ones(static_tensor.shape[0], dtype=torch.bool)
    best_loss = float('inf')
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for inputs, labels in loader(x_train, y_train, batch_size=batch_size, shuffle=True, seed=seed):
            optimizer.zero_grad(set_to_none=True)
            preds = model(inputs.float(), static_tensor, adj_tensor)
            loss = masked_huber(preds, labels.float(), mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.item()) * len(labels)
            train_count += len(labels)
        train_loss = train_total / max(train_count, 1)
        val_loss = validation_loss(model, x_val, y_val, static_tensor, adj_tensor, batch_size, mask)
        history.append({"epoch": float(epoch), "training_loss": float(train_loss), "validation_loss": float(val_loss)})
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError('training did not produce a valid checkpoint')
    model.load_state_dict(best_state)
    return model, history


def predict_target(model: nn.Module, x_pred: np.ndarray, *, static_tensor: torch.Tensor, adj_tensor: torch.Tensor, batch_size: int, target_index: int) -> np.ndarray:
    model.eval()
    outputs = []
    loader_obj = DataLoader(TensorDataset(torch.from_numpy(x_pred)), batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for (inputs,) in loader_obj:
            preds = model(inputs.float(), static_tensor, adj_tensor)
            outputs.append(preds[:, target_index].cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def compare_against(candidate: pd.DataFrame, baseline: pd.DataFrame, baseline_column: str) -> tuple[float, float, int, int, int]:
    merged = baseline.merge(candidate, on=["target_station_id", "event_id"], how="inner", validate="one_to_one")
    merged["gain_pct"] = 100.0 * (merged[baseline_column] - merged["candidate_rmse"]) / merged[baseline_column]
    diffs = merged["candidate_rmse"] - merged[baseline_column]
    nonzero = diffs[diffs != 0.0]
    p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 5 else float('nan')
    wins = int((merged["gain_pct"] > 5.0).sum())
    ties = int(((merged["gain_pct"] >= -5.0) & (merged["gain_pct"] <= 5.0)).sum())
    losses = int((merged["gain_pct"] < -5.0).sum())
    return float(merged["gain_pct"].mean()), p_value, wins, ties, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=Path, default=ROOT / 'configs' / 'experiments' / 'seven_station_56.yaml')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'backbone_followups' / 'stgcn_lite_017_pilot')
    parser.add_argument('--seed', type=int, default=20260723)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-epochs', type=int, default=20)
    args = parser.parse_args()

    config, data, coastal = load_data(args.config_path)
    feature_columns = list(config['features']['dynamic'])
    lookback = int(config['model']['lookback_hours'])
    batch_size = int(config['model']['batch_size'])
    learning_rate = float(config['model']['learning_rate'])
    patience = int(config['model']['early_stopping_patience'])
    tasks = build_task_grid(config)

    output_dir = args.output_dir
    pred_dir = output_dir / 'predictions'
    hist_dir = output_dir / 'training_history'
    metrics_dir = output_dir / 'metrics'
    for directory in [pred_dir, hist_dir, metrics_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    task_rows = []
    manifest_tasks = []
    started = datetime.now(timezone.utc)

    for index, task in enumerate(tasks, start=1):
        target = str(task['target_station_id'])
        test_event = str(task['test_event'])
        validation_event = str(task['validation_event'])
        train_events = list(task['train_events'])
        source_pool = list(task['source_station_ids'])
        training_rows = data.loc[data['station_id'].isin(source_pool) & data['event_id'].isin(train_events)].copy()
        scalers = fit_scalers(training_rows, feature_columns)

        x_train, y_train, _ = build_graph_windows(data, station_ids=source_pool, event_ids=train_events, feature_columns=feature_columns, lookback=lookback, scalers=scalers, include_labels=True)
        x_val, y_val, _ = build_graph_windows(data, station_ids=source_pool, event_ids=[validation_event], feature_columns=feature_columns, lookback=lookback, scalers=scalers, include_labels=True)
        static_train, adj_train = build_static_inputs(coastal, data, station_ids=source_pool, scalers=scalers)
        model, history = train_task_model(
            x_train, y_train, x_val, y_val,
            static_tensor=torch.from_numpy(static_train).float(),
            adj_tensor=torch.from_numpy(adj_train).float(),
            dynamic_in_dim=len(feature_columns),
            static_in_dim=len(COASTAL_COLUMNS),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=int(args.max_epochs),
            patience=patience,
            seed=int(args.seed),
        )
        pd.DataFrame(history).to_csv(hist_dir / f'target_{target}__event_{test_event}__seed_{int(args.seed)}.csv', index=False)

        predict_nodes = source_pool + [target]
        x_pred, _, meta = build_graph_windows(data, station_ids=predict_nodes, event_ids=[test_event], feature_columns=feature_columns, lookback=lookback, scalers=scalers, include_labels=False)
        static_pred, adj_pred = build_static_inputs(coastal, data, station_ids=predict_nodes, scalers=scalers)
        residual_scaled = predict_target(
            model,
            x_pred,
            static_tensor=torch.from_numpy(static_pred).float(),
            adj_tensor=torch.from_numpy(adj_pred).float(),
            batch_size=batch_size,
            target_index=len(predict_nodes) - 1,
        )
        residual = scalers.label_scaler.inverse_transform(residual_scaled.reshape(-1, 1)).ravel()
        target_rows = data.loc[(data['station_id'] == target) & (data['event_id'] == test_event)].sort_values('timestamp_utc').copy()
        target_rows = target_rows.set_index('timestamp_utc').loc[pd.DatetimeIndex(meta['timestamp_utc'])].reset_index()
        predicted_water = residual + target_rows['predicted_tide_m'].to_numpy(float)
        pred_df = pd.DataFrame({
            'timestamp_utc': meta['timestamp_utc'],
            'predicted_ntr_m': residual,
            'predicted_water_level_m': predicted_water,
        })
        pred_df.to_csv(pred_dir / f'target_{target}__event_{test_event}__seed_{int(args.seed)}.csv', index=False)

        metrics = evaluate_event(target_rows['water_level_m'], predicted_water, target_rows['timestamp_utc'])
        seed_row = {
            'target_station_id': target,
            'event_id': test_event,
            'seed': int(args.seed),
            'method': 'stgcn_lite',
            **metrics,
        }
        seed_rows.append(seed_row)
        task_rows.append({
            'target_station_id': target,
            'event_id': test_event,
            'candidate_rmse': float(metrics['rmse']),
        })
        manifest_tasks.append({
            'task_id': task['task_id'],
            'target_station_id': target,
            'test_event': test_event,
            'validation_event': validation_event,
            'train_events': train_events,
            'source_pool_size': len(source_pool),
            'epochs_ran': len(history),
            'best_validation_loss': min(row['validation_loss'] for row in history),
        })
        print(f"[stgcn_lite_017_pilot] finished {index}/{len(tasks)} tasks", flush=True)

    seed_df = pd.DataFrame(seed_rows)
    task_df = pd.DataFrame(task_rows)
    seed_df.to_csv(metrics_dir / 'seed_level_metrics.csv', index=False)
    task_df.to_csv(metrics_dir / 'task_level_rmse.csv', index=False)

    baseline_path = ROOT / 'results' / 'manual_followups' / '2026-08-03_batch_012_cosine' / '..' / '..' / '..' / 'feature_group_ablation_subset_optimized' / 'per_task_metrics.csv'
    baseline_path = baseline_path.resolve()
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[['target_station_id', 'event_id', 'config', 'rmse']].copy()
    baseline['target_station_id'] = baseline['target_station_id'].astype(str)
    wave_baseline = baseline.loc[baseline['config'] == 'wave_coastal', ['target_station_id', 'event_id', 'rmse']].rename(columns={'rmse': 'wave_coastal_rmse'})
    full_baseline = baseline.loc[baseline['config'] == 'full_sasc', ['target_station_id', 'event_id', 'rmse']].rename(columns={'rmse': 'full_sasc_rmse'})

    wave_gain, wave_p, wave_w, wave_t, wave_l = compare_against(task_df, wave_baseline, 'wave_coastal_rmse')
    full_gain, full_p, full_w, full_t, full_l = compare_against(task_df, full_baseline, 'full_sasc_rmse')
    summary = pd.DataFrame([{
        'stage': 'stgcn_lite_017_pilot',
        'seed': int(args.seed),
        'rmse_mean': float(task_df['candidate_rmse'].mean()),
        'hidden_dim': int(args.hidden_dim),
        'dropout': float(args.dropout),
        'max_epochs': int(args.max_epochs),
        'gain_pct_vs_wave_coastal': wave_gain,
        'p_vs_wave_coastal': wave_p,
        'win_vs_wave_coastal': wave_w,
        'tie_vs_wave_coastal': wave_t,
        'loss_vs_wave_coastal': wave_l,
        'gain_pct_vs_full_sasc': full_gain,
        'p_vs_full_sasc': full_p,
        'win_vs_full_sasc': full_w,
        'tie_vs_full_sasc': full_t,
        'loss_vs_full_sasc': full_l,
    }])
    summary.to_csv(metrics_dir / 'summary.csv', index=False)
    (metrics_dir / 'summary.md').write_text(
        '\n'.join([
            '# ST-GCN-lite 017 pilot summary',
            '',
            '- source-only train on six-node graphs',
            '- active target inserted only at prediction time',
            '- one-step graph windows with lookback=24 hours',
            '',
            '```text',
            summary.to_string(index=False, justify='center', float_format=lambda x: f'{x:.4f}'),
            '```',
        ]),
        encoding='utf-8',
    )
    manifest = {
        'stage_name': 'stgcn_lite_017_pilot',
        'started_utc': started.isoformat(),
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'config_path': str(args.config_path),
        'config_sha256': hashlib.sha256(args.config_path.read_text(encoding='utf-8').encode('utf-8')).hexdigest(),
        'target_labels_used_for_training': False,
        'target_labels_used_for_validation': False,
        'target_labels_used_for_prediction': False,
        'seed': int(args.seed),
        'hidden_dim': int(args.hidden_dim),
        'dropout': float(args.dropout),
        'max_epochs': int(args.max_epochs),
        'task_count': len(tasks),
        'tasks': manifest_tasks,
    }
    (output_dir / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.to_csv(index=False))
    print(f'saved: {metrics_dir / "summary.csv"}')
    print(f'saved: {metrics_dir / "summary.md"}')


if __name__ == '__main__':
    main()
