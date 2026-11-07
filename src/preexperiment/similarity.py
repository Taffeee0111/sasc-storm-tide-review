"""Label-free station/event descriptor distances and similarities."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


STATISTICS = ("mean", "std", "min", "max")
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    value = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


def event_descriptors(frame: pd.DataFrame, variables: list[str]) -> pd.Series:
    working: dict[str, pd.Series] = {}
    for variable in variables:
        values = pd.to_numeric(frame[variable], errors="coerce")
        if variable == "mwd":
            radians = np.deg2rad(values % 360.0)
            working["mwd_sin"] = pd.Series(np.sin(radians), index=frame.index)
            working["mwd_cos"] = pd.Series(np.cos(radians), index=frame.index)
        else:
            working[variable] = values

    result: dict[str, float] = {}
    for name, values in working.items():
        clean = values.dropna().astype(float)
        if clean.empty:
            raise ValueError(f"descriptor variable has no finite values: {name}")
        result[f"{name}__mean"] = float(clean.mean())
        result[f"{name}__std"] = float(clean.std(ddof=0))
        result[f"{name}__min"] = float(clean.min())
        result[f"{name}__max"] = float(clean.max())
        for quantile in QUANTILES:
            result[f"{name}__q{int(quantile * 100):02d}"] = float(clean.quantile(quantile))
    return pd.Series(result, dtype=float)


def standardized_distance(
    source: pd.Series, target: pd.Series, reference: pd.DataFrame
) -> float:
    columns = list(source.index)
    if list(target.index) != columns:
        target = target.reindex(columns)
    reference = reference.reindex(columns=columns).astype(float)
    means = reference.mean(axis=0)
    scales = reference.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    source_z = (source.astype(float) - means) / scales
    target_z = (target.astype(float) - means) / scales
    difference = (source_z - target_z).to_numpy(dtype=float)
    if not np.isfinite(difference).all():
        raise ValueError("standardized distance contains non-finite descriptors")
    return float(np.sqrt(np.mean(difference**2)))


def softmax_similarity(distances, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(distances, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("distances must be a finite non-empty vector")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = -values / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def zscore_distances(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    scale = float(values.std(ddof=0))
    if scale == 0 or not np.isfinite(scale):
        return pd.Series(0.0, index=values.index)
    return (values - float(values.mean())) / scale
