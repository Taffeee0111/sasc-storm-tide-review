"""Event-level forecast metrics used by the feasibility gate."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _one_dimensional_finite(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def evaluate_event(y_true, y_pred, timestamps) -> dict[str, float]:
    """Evaluate paired event observations and forecasts.

    Peak errors are signed predicted-minus-observed differences: positive means a
    high or late forecast and negative means a low or early forecast. The first
    occurrence is used when a series has tied peaks. KGE is undefined when either
    standard deviation or the observed mean is zero; NSE is undefined when
    observations have zero variance. kge_r/kge_alpha/kge_beta are the three
    Gupta et al. (2009) KGE sub-components (correlation, variability ratio, bias
    ratio) reported separately so a KGE shortfall can be attributed to a specific
    cause. pbias follows the Moriasi et al. (2007) sign convention (positive =
    model underestimation, negative = overestimation). rsr is RMSE normalized by
    the observed standard deviation, and willmott_d is the original Willmott
    (1981) index of agreement. Undefined scores are returned as ``NaN`` while the
    remaining metrics are still reported.
    """
    observed = _one_dimensional_finite(y_true, "y_true")
    predicted = _one_dimensional_finite(y_pred, "y_pred")
    try:
        times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="raise"))
    except (TypeError, ValueError) as error:
        raise ValueError("timestamps must contain valid datetimes") from error
    if times.hasnans:
        raise ValueError("timestamps must contain valid datetimes")

    if not len(observed):
        raise ValueError("evaluation requires at least one observation")
    if len(observed) != len(predicted) or len(observed) != len(times):
        raise ValueError("y_true, y_pred, and timestamps must have the same length")
    if times.has_duplicates:
        raise ValueError("timestamps must be unique")

    residual = predicted - observed
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))

    observed_mean = float(np.mean(observed))
    observed_std = float(np.std(observed))
    predicted_std = float(np.std(predicted))
    if len(observed) < 2 or observed_std == 0.0 or predicted_std == 0.0 or observed_mean == 0.0:
        kge = float("nan")
        kge_r = float("nan")
        kge_alpha = float("nan")
        kge_beta = float("nan")
    else:
        correlation = float(np.corrcoef(observed, predicted)[0, 1])
        variability_ratio = predicted_std / observed_std
        bias_ratio = float(np.mean(predicted)) / observed_mean
        kge = float(
            1.0
            - np.sqrt(
                (correlation - 1.0) ** 2
                + (variability_ratio - 1.0) ** 2
                + (bias_ratio - 1.0) ** 2
            )
        )
        kge_r = correlation
        kge_alpha = variability_ratio
        kge_beta = bias_ratio

    denominator = float(np.sum((observed - observed_mean) ** 2))
    nse = float("nan") if denominator == 0.0 else float(1.0 - np.sum(residual**2) / denominator)

    observed_sum = float(np.sum(observed))
    pbias = (
        float("nan")
        if observed_sum == 0.0
        else float(100.0 * np.sum(observed - predicted) / observed_sum)
    )
    rsr = float("nan") if observed_std == 0.0 else float(rmse / observed_std)

    willmott_denominator = float(
        np.sum(
            (np.abs(predicted - observed_mean) + np.abs(observed - observed_mean)) ** 2
        )
    )
    willmott_d = (
        float("nan")
        if willmott_denominator == 0.0
        else float(1.0 - np.sum(residual**2) / willmott_denominator)
    )

    observed_peak_index = int(np.argmax(observed))
    predicted_peak_index = int(np.argmax(predicted))
    peak_level_error = float(
        predicted[predicted_peak_index] - observed[observed_peak_index]
    )
    peak_delta = times[predicted_peak_index] - times[observed_peak_index]
    peak_time_error = float(peak_delta.total_seconds() / 3600.0)

    return {
        "rmse": rmse,
        "mae": mae,
        "kge": kge,
        "kge_r": kge_r,
        "kge_alpha": kge_alpha,
        "kge_beta": kge_beta,
        "nse": nse,
        "pbias": pbias,
        "rsr": rsr,
        "willmott_d": willmott_d,
        "peak_level_error_m": peak_level_error,
        "peak_time_error_h": peak_time_error,
    }


def evaluate_threshold_exceedance(
    y_true, y_pred, *, top_percentile: float = 0.90
) -> dict[str, float]:
    """Categorical exceedance verification (NWS/NOAA-style POD/FAR/CSI) for one
    event. The threshold is the given percentile of the OBSERVED series for
    this event (default: top 10%), matching the extreme-value stratification
    convention used elsewhere in this project. A timestep counts as a "hit" if
    both observed and predicted exceed the threshold, a "miss" if only
    observed exceeds it, and a "false alarm" if only predicted exceeds it.
    POD/FAR/CSI are undefined (NaN) when there are zero observed exceedances
    (no positive class to detect).
    """
    observed = _one_dimensional_finite(y_true, "y_true")
    predicted = _one_dimensional_finite(y_pred, "y_pred")
    if len(observed) != len(predicted):
        raise ValueError("y_true and y_pred must have the same length")
    if not len(observed):
        raise ValueError("evaluation requires at least one observation")
    if not 0.0 < top_percentile < 1.0:
        raise ValueError("top_percentile must be within (0, 1)")

    threshold = float(np.quantile(observed, top_percentile))
    observed_exceeds = observed >= threshold
    predicted_exceeds = predicted >= threshold

    hits = int(np.sum(observed_exceeds & predicted_exceeds))
    misses = int(np.sum(observed_exceeds & ~predicted_exceeds))
    false_alarms = int(np.sum(~observed_exceeds & predicted_exceeds))

    if hits + misses == 0:
        pod = float("nan")
        csi = float("nan")
    else:
        pod = float(hits / (hits + misses))
        csi = float(hits / (hits + misses + false_alarms))
    far = float("nan") if hits + false_alarms == 0 else float(false_alarms / (hits + false_alarms))

    return {
        "threshold": threshold,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "pod": pod,
        "far": far,
        "csi": csi,
    }
