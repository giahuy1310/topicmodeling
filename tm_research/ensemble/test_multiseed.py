"""Unit tests for multi-seed QLoRA metric aggregation.

Run with:  python -m pytest tm_research/ensemble/test_multiseed.py -v
"""

from __future__ import annotations

import pytest

from tm_research.ensemble.multiseed import summarize_seed_metrics

# Student's t 0.975 quantile, df=4 (five seeds). Independent of the module under test.
_T_CRIT_DF4 = 2.7764451051977987


def test_summarize_seed_metrics_mean_sample_std_and_95pct_t_ci():
    runs = [
        {"seed": 42, "accuracy": 0.70, "weighted_f1": 0.70, "macro_f1": 0.60},
        {"seed": 123, "accuracy": 0.72, "weighted_f1": 0.71, "macro_f1": 0.62},
        {"seed": 2024, "accuracy": 0.68, "weighted_f1": 0.69, "macro_f1": 0.58},
        {"seed": 7, "accuracy": 0.74, "weighted_f1": 0.72, "macro_f1": 0.64},
        {"seed": 99, "accuracy": 0.66, "weighted_f1": 0.68, "macro_f1": 0.56},
    ]
    agg = summarize_seed_metrics(runs)

    n = 5
    w_mean = 0.70
    w_std = 0.015811388300841896  # sample std of {0.70,0.71,0.69,0.72,0.68}
    w_half = _T_CRIT_DF4 * (w_std / (n ** 0.5))

    a_mean = 0.70
    a_std = 0.03162277660168379  # sample std of {0.70,0.72,0.68,0.74,0.66}
    a_half = _T_CRIT_DF4 * (a_std / (n ** 0.5))

    m_mean = 0.60
    m_std = a_std  # same spread as accuracy, shifted
    m_half = a_half

    for key, mean, std, half in (
        ("weighted_f1", w_mean, w_std, w_half),
        ("accuracy", a_mean, a_std, a_half),
        ("macro_f1", m_mean, m_std, m_half),
    ):
        row = agg[key]
        assert row["n"] == n
        assert row["mean"] == pytest.approx(mean)
        assert row["std"] == pytest.approx(std)
        assert row["ci95"][0] == pytest.approx(mean - half)
        assert row["ci95"][1] == pytest.approx(mean + half)
        assert row["ci95"][0] < row["mean"] < row["ci95"][1]


def test_summarize_seed_metrics_empty_runs_raises():
    with pytest.raises(ValueError, match="empty"):
        summarize_seed_metrics([])
