"""Unit tests for multi-seed artifact paths and validation and test averages.

Run with:  python -m pytest tm_research/ensemble/test_multiseed.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tm_research.ensemble import utils_io
from tm_research.ensemble.multiseed import (
    PIPELINE_SEEDS,
    collect_test_runs,
    collect_val_runs,
    resolve_pipeline_seeds,
    seed_is_complete,
    summarize_seed_metrics,
    write_test_multiseed_summary,
    write_val_multiseed_summary,
)

# Student's t 0.975 quantile, df=4 (five seeds). Independent of the module under test.
_T_CRIT_DF4 = 2.7764451051977987

_DIR_ATTRS = (
    "ARTIFACTS_DIR",
    "PROBS_DIR",
    "META_JSONL_DIR",
    "METRICS_DIR",
    "LORA_DIR",
    "_SESSION_ARTIFACTS_DIR",
    "PERSISTENT_ARTIFACTS_DIR",
)

_VAL_KEYS = ("accuracy", "macro_f1", "weighted_f1")


def _capture_io_state() -> dict:
    return {name: getattr(utils_io, name) for name in _DIR_ATTRS}


def _restore_io_state(saved: dict) -> None:
    for name, value in saved.items():
        setattr(utils_io, name, value)


@pytest.fixture(autouse=True)
def _isolate_artifact_dirs():
    saved = _capture_io_state()
    try:
        yield
    finally:
        _restore_io_state(saved)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def test_resolve_pipeline_seeds_default(monkeypatch):
    monkeypatch.delenv("TM_RUN_SEEDS", raising=False)
    assert PIPELINE_SEEDS == (42, 131, 9998)
    assert resolve_pipeline_seeds() == PIPELINE_SEEDS


def test_resolve_pipeline_seeds_blank_env_uses_default(monkeypatch):
    monkeypatch.setenv("TM_RUN_SEEDS", "   ")
    assert resolve_pipeline_seeds() == PIPELINE_SEEDS


def test_resolve_pipeline_seeds_override(monkeypatch):
    monkeypatch.setenv("TM_RUN_SEEDS", "42")
    assert resolve_pipeline_seeds() == (42,)
    monkeypatch.setenv("TM_RUN_SEEDS", "42, 131")
    assert resolve_pipeline_seeds() == (42, 131)


def test_resolve_pipeline_seeds_rejects_blank_entries(monkeypatch):
    monkeypatch.setenv("TM_RUN_SEEDS", ",")
    with pytest.raises(ValueError):
        resolve_pipeline_seeds()
    monkeypatch.setenv("TM_RUN_SEEDS", "42,,131")
    with pytest.raises(ValueError):
        resolve_pipeline_seeds()


def test_seed_is_complete_without_activating(tmp_path):
    before = utils_io.ARTIFACTS_DIR
    rels = ("probs/phobert_val.npy", "metrics/phobert.json")
    metrics = tmp_path / "seeds" / "42" / "metrics"
    probs = tmp_path / "seeds" / "42" / "probs"
    metrics.mkdir(parents=True)
    probs.mkdir()
    (metrics / "phobert.json").write_text("{}", encoding="utf-8")

    assert seed_is_complete(42, rels, root=tmp_path) is False
    (probs / "phobert_val.npy").write_bytes(b"\x00")
    assert seed_is_complete(42, rels, root=tmp_path) is True
    assert seed_is_complete(131, rels, root=tmp_path) is False
    assert utils_io.ARTIFACTS_DIR == before


def test_activate_seed_artifacts_and_reset(tmp_path):
    before = tuple(getattr(utils_io, name) for name in _DIR_ATTRS[:5])
    session_seed = utils_io._SESSION_ARTIFACTS_DIR / "seeds" / "42"
    session_existed = session_seed.exists()

    seed_dir = utils_io.activate_seed_artifacts(42, root=tmp_path)
    try:
        assert seed_dir == tmp_path / "seeds" / "42"
        assert utils_io.ARTIFACTS_DIR == seed_dir
        assert utils_io.PROBS_DIR == seed_dir / "probs"
        assert utils_io.META_JSONL_DIR == seed_dir / "meta_jsonl"
        assert utils_io.METRICS_DIR == seed_dir / "metrics"
        assert utils_io.LORA_DIR == seed_dir / "lora_adapter"
        for directory in (
            utils_io.ARTIFACTS_DIR,
            utils_io.PROBS_DIR,
            utils_io.META_JSONL_DIR,
            utils_io.METRICS_DIR,
            utils_io.LORA_DIR,
        ):
            assert directory.is_dir()

        probs = np.array([[0.1, 0.9], [0.4, 0.6]], dtype=np.float32)
        saved_probs = utils_io.save_probs("phobert", "val", probs)
        assert saved_probs == seed_dir / "probs" / "phobert_val.npy"
        loaded = utils_io.load_probs("phobert", "val")
        assert loaded.shape == (2, 2)
        assert loaded.dtype == np.float32

        saved_metrics = utils_io.save_metrics("phobert", {"val_accuracy": 0.5})
        assert saved_metrics == seed_dir / "metrics" / "phobert.json"
        assert utils_io.load_metrics("phobert")["val_accuracy"] == 0.5
    finally:
        restored = utils_io.reset_artifacts_root()

    assert restored == utils_io._SESSION_ARTIFACTS_DIR
    assert tuple(getattr(utils_io, name) for name in _DIR_ATTRS[:5]) == before
    assert session_seed.exists() == session_existed


def test_push_and_pull_copy_session_root_while_seed_is_active(tmp_path):
    session = tmp_path / "session"
    persistent = tmp_path / "persistent"
    session.mkdir()
    (session / "root_marker.txt").write_text("session-root", encoding="utf-8")
    utils_io._SESSION_ARTIFACTS_DIR = session
    utils_io.PERSISTENT_ARTIFACTS_DIR = persistent

    seed_dir = utils_io.activate_seed_artifacts(42)
    (seed_dir / "seed_only.txt").write_text("inside-seed", encoding="utf-8")
    assert utils_io.ARTIFACTS_DIR == seed_dir

    pushed = utils_io.push_artifacts_to_persistent(verbose=False)
    assert pushed == persistent
    assert (persistent / "root_marker.txt").read_text(encoding="utf-8") == "session-root"
    assert (persistent / "seeds" / "42" / "seed_only.txt").read_text(encoding="utf-8") == "inside-seed"
    assert not (persistent / "seed_only.txt").exists()
    assert not (persistent / "probs").exists()

    (persistent / "from_drive.txt").write_text("drive", encoding="utf-8")
    pulled = utils_io.pull_artifacts_from_persistent(verbose=False)
    assert pulled == session
    assert (session / "from_drive.txt").read_text(encoding="utf-8") == "drive"
    assert not (seed_dir / "from_drive.txt").exists()


def test_push_and_pull_noop_when_session_root_is_persistent(tmp_path):
    root = tmp_path / "same"
    root.mkdir()
    (root / "keep.txt").write_text("keep", encoding="utf-8")
    utils_io._SESSION_ARTIFACTS_DIR = root
    utils_io.PERSISTENT_ARTIFACTS_DIR = root
    seed_dir = utils_io.activate_seed_artifacts(42)
    (seed_dir / "seed_only.txt").write_text("inside-seed", encoding="utf-8")

    assert utils_io.push_artifacts_to_persistent(verbose=False) is None
    assert utils_io.pull_artifacts_from_persistent(verbose=False) is None
    assert not (root / "seed_only.txt").exists()
    assert (root / "keep.txt").read_text(encoding="utf-8") == "keep"


def _build_val_tree(root: Path) -> None:
    ensemble = {
        42: {
            "lora_gemma_meta": {"accuracy": 0.50, "macro_f1": 0.40, "weighted_f1": 0.55, "extra": 1},
            "weighted_soft_voting": {"accuracy": 0.40, "macro_f1": 0.30, "weighted_f1": 0.35},
            "zero_shot_gemma": {"accuracy": 0.20, "macro_f1": 0.10, "weighted_f1": 0.15},
            "phobert": {"accuracy": 0.99, "macro_f1": 0.99, "weighted_f1": 0.99},
        },
        131: {
            "lora_gemma_meta": {"accuracy": 0.60, "macro_f1": 0.50, "weighted_f1": 0.65},
            "weighted_soft_voting": {"accuracy": 0.50, "macro_f1": 0.40, "weighted_f1": 0.45},
            "zero_shot_gemma": {"accuracy": 0.30, "macro_f1": 0.20, "weighted_f1": 0.25},
        },
        9998: {
            "lora_gemma_meta": {"accuracy": 0.70, "macro_f1": 0.60, "weighted_f1": 0.75},
            "zero_shot_gemma": {"accuracy": 0.40, "macro_f1": 0.30, "weighted_f1": 0.35},
        },
    }
    standalone = {
        "cafebert_standalone": {
            42: {"val_accuracy": 0.80, "val_f1": 0.70, "val_f1_weighted": 0.75, "seed": 0, "test_accuracy": 0.01},
            131: {"val_accuracy": 0.82, "val_f1": 0.72, "val_f1_weighted": 0.77},
            9998: {"val_accuracy": 0.84, "val_f1": 0.74, "val_f1_weighted": 0.79},
        },
        "visobert_standalone": {
            42: {"val_accuracy": 0.10, "val_f1": 0.20, "val_f1_weighted": 0.30},
            131: {"val_accuracy": 0.30, "val_f1": 0.40, "val_f1_weighted": 0.50},
        },
    }
    for seed, payload in ensemble.items():
        _write_json(root / "seeds" / str(seed) / "metrics" / "val_systems.json", payload)
    for name, by_seed in standalone.items():
        for seed, payload in by_seed.items():
            _write_json(root / "seeds" / str(seed) / "metrics" / f"{name}.json", payload)


def _assert_aggregate_matches(summary: dict, system: str) -> None:
    expected = summarize_seed_metrics(summary["runs"][system], keys=_VAL_KEYS)
    got = summary["aggregate"][system]
    assert list(got) == list(_VAL_KEYS)
    for key in _VAL_KEYS:
        assert got[key]["n"] == expected[key]["n"]
        assert got[key]["mean"] == pytest.approx(expected[key]["mean"])
        assert got[key]["std"] == pytest.approx(expected[key]["std"])
        assert got[key]["ci95"] == pytest.approx(expected[key]["ci95"])


def test_write_val_multiseed_summary_means_and_standalone_keys(tmp_path):
    _build_val_tree(tmp_path)
    out = write_val_multiseed_summary(seeds=(42, 131, 9998), root=tmp_path)

    assert out == tmp_path / "metrics" / "val_multiseed_summary.json"
    assert out.is_file()
    assert not (tmp_path / "seeds" / "42" / "metrics" / "val_multiseed_summary.json").exists()

    summary = json.loads(out.read_text(encoding="utf-8"))
    assert set(summary) == {"seeds", "runs", "aggregate"}
    assert summary["seeds"] == [42, 131, 9998]

    runs = summary["runs"]
    assert list(runs) == [
        "lora_gemma_meta",
        "weighted_soft_voting",
        "zero_shot_gemma",
        "cafebert_standalone",
        "visobert_standalone",
    ]
    assert "logreg_meta" not in runs
    assert "logreg_meta" not in summary["aggregate"]
    assert "phobert" not in runs

    assert runs["cafebert_standalone"][0] == {
        "seed": 42,
        "accuracy": 0.80,
        "macro_f1": 0.70,
        "weighted_f1": 0.75,
    }
    assert [row["seed"] for row in runs["cafebert_standalone"]] == [42, 131, 9998]
    assert [row["seed"] for row in runs["weighted_soft_voting"]] == [42, 131]
    assert [row["seed"] for row in runs["visobert_standalone"]] == [42, 131]
    assert set(runs["lora_gemma_meta"][0]) == {"seed", "accuracy", "macro_f1", "weighted_f1"}

    assert summary["aggregate"]["lora_gemma_meta"]["accuracy"]["mean"] == pytest.approx(0.60)
    assert summary["aggregate"]["lora_gemma_meta"]["accuracy"]["n"] == 3
    assert summary["aggregate"]["cafebert_standalone"]["accuracy"]["mean"] == pytest.approx(0.82)
    assert summary["aggregate"]["cafebert_standalone"]["macro_f1"]["mean"] == pytest.approx(0.72)
    assert summary["aggregate"]["cafebert_standalone"]["weighted_f1"]["mean"] == pytest.approx(0.77)
    assert summary["aggregate"]["weighted_soft_voting"]["accuracy"]["n"] == 2
    assert summary["aggregate"]["weighted_soft_voting"]["accuracy"]["mean"] == pytest.approx(0.45)
    assert summary["aggregate"]["visobert_standalone"]["accuracy"]["mean"] == pytest.approx(0.20)
    assert summary["aggregate"]["visobert_standalone"]["accuracy"]["n"] == 2

    for system in runs:
        _assert_aggregate_matches(summary, system)

    collected = collect_val_runs(seeds=(42, 131, 9998), root=tmp_path)
    assert collected == runs


def test_write_val_multiseed_summary_uses_session_root_not_active_seed(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    _build_val_tree(session)
    monkeypatch.delenv("TM_RUN_SEEDS", raising=False)
    utils_io._SESSION_ARTIFACTS_DIR = session
    utils_io.activate_seed_artifacts(42)

    out = write_val_multiseed_summary()
    assert out == session / "metrics" / "val_multiseed_summary.json"
    assert not (session / "seeds" / "42" / "metrics" / "val_multiseed_summary.json").exists()
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["seeds"] == [42, 131, 9998]
    assert "cafebert_standalone" in summary["runs"]


def _build_test_tree(root: Path) -> None:
    ensemble = {
        42: {
            "lora_gemma_meta": {"accuracy": 0.50, "macro_f1": 0.40, "weighted_f1": 0.55, "extra": 1},
            "weighted_soft_voting": {"accuracy": 0.40, "macro_f1": 0.30, "weighted_f1": 0.35},
            "logreg_meta": {"accuracy": 0.45, "macro_f1": 0.35, "weighted_f1": 0.42},
            "zero_shot_gemma": {"accuracy": 0.20, "macro_f1": 0.10, "weighted_f1": 0.15},
            "phobert": {"accuracy": 0.99, "macro_f1": 0.99, "weighted_f1": 0.99},
        },
        131: {
            "lora_gemma_meta": {"accuracy": 0.60, "macro_f1": 0.50, "weighted_f1": 0.65},
            "weighted_soft_voting": {"accuracy": 0.50, "macro_f1": 0.40, "weighted_f1": 0.45},
            "logreg_meta": {"accuracy": 0.55, "macro_f1": 0.45, "weighted_f1": 0.52},
            "zero_shot_gemma": {"accuracy": 0.30, "macro_f1": 0.20, "weighted_f1": 0.25},
        },
        9998: {
            "lora_gemma_meta": {"accuracy": 0.70, "macro_f1": 0.60, "weighted_f1": 0.75},
            "zero_shot_gemma": {"accuracy": 0.40, "macro_f1": 0.30, "weighted_f1": 0.35},
        },
    }
    for seed, payload in ensemble.items():
        _write_json(root / "seeds" / str(seed) / "metrics" / "test_systems.json", payload)
        _write_json(
            root / "seeds" / str(seed) / "metrics" / "val_systems.json",
            {"lora_gemma_meta": {"accuracy": 0.01, "macro_f1": 0.01, "weighted_f1": 0.01}},
        )


def test_write_test_multiseed_summary_mean_and_spread(tmp_path):
    _build_test_tree(tmp_path)
    out = write_test_multiseed_summary(seeds=(42, 131, 9998), root=tmp_path)

    assert out == tmp_path / "metrics" / "test_multiseed_summary.json"
    assert out.is_file()
    assert not (tmp_path / "seeds" / "42" / "metrics" / "test_multiseed_summary.json").exists()

    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["seeds"] == [42, 131, 9998]
    runs = summary["runs"]
    assert list(runs) == [
        "lora_gemma_meta",
        "weighted_soft_voting",
        "logreg_meta",
        "zero_shot_gemma",
    ]
    assert "phobert" not in runs
    assert "cafebert_standalone" not in runs
    assert [row["seed"] for row in runs["lora_gemma_meta"]] == [42, 131, 9998]
    assert [row["seed"] for row in runs["logreg_meta"]] == [42, 131]
    assert runs["lora_gemma_meta"][0]["accuracy"] == pytest.approx(0.50)
    assert summary["aggregate"]["lora_gemma_meta"]["accuracy"]["mean"] == pytest.approx(0.60)
    assert summary["aggregate"]["lora_gemma_meta"]["accuracy"]["n"] == 3
    assert summary["aggregate"]["logreg_meta"]["accuracy"]["n"] == 2
    for system in runs:
        _assert_aggregate_matches(summary, system)

    collected = collect_test_runs(seeds=(42, 131, 9998), root=tmp_path)
    assert collected == runs
    assert collect_val_runs(seeds=(42, 131, 9998), root=tmp_path)["lora_gemma_meta"][0]["accuracy"] == pytest.approx(0.01)
