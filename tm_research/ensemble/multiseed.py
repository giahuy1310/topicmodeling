"""Multi-seed metric aggregation for QLoRA Gemma statistical validation."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

DEFAULT_METRIC_KEYS = ("accuracy", "weighted_f1", "macro_f1")


def summarize_seed_metrics(
    runs: Sequence[Mapping],
    keys: Iterable[str] = DEFAULT_METRIC_KEYS,
) -> dict:
    """Mean, sample std (ddof=1), and 95% Student-t CI for each metric.

    ``runs`` is a list of dicts each containing the keys in ``keys``.
    Returns ``{metric: {"mean", "std", "ci95": [lo, hi], "n"}}``.
    """

    runs = list(runs)
    n = len(runs)
    if n == 0:
        raise ValueError("runs must be non-empty")

    out: Dict[str, dict] = {}
    for key in keys:
        values = np.asarray([float(r[key]) for r in runs], dtype=np.float64)
        mean = float(values.mean())
        if n == 1:
            out[key] = {"mean": mean, "std": 0.0, "ci95": [mean, mean], "n": n}
            continue
        std = float(values.std(ddof=1))
        se = std / np.sqrt(n)
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        half = t_crit * se
        out[key] = {
            "mean": mean,
            "std": std,
            "ci95": [float(mean - half), float(mean + half)],
            "n": n,
        }
    return out
