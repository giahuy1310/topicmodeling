"""Multi-seed metric aggregation for validation and test summaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

from tm_research.ensemble import utils_io

DEFAULT_METRIC_KEYS = ("accuracy", "weighted_f1", "macro_f1")
VAL_METRIC_KEYS = ("accuracy", "macro_f1", "weighted_f1")

PIPELINE_SEEDS = (42, 131, 9998)

# Four ensemble systems scored into metrics/val_systems.json by notebook 07.
ENSEMBLE_VAL_SYSTEMS = (
    "lora_gemma_meta",
    "weighted_soft_voting",
    "logreg_meta",
    "zero_shot_gemma",
)
# Notebooks 09 and 10. Their JSON already uses val_* keys.
STANDALONE_VAL_SYSTEMS = (
    "cafebert_standalone",
    "visobert_standalone",
)
_ENSEMBLE_KEY_MAP = {
    "accuracy": "accuracy",
    "macro_f1": "macro_f1",
    "weighted_f1": "weighted_f1",
}
_STANDALONE_KEY_MAP = {
    "val_accuracy": "accuracy",
    "val_f1": "macro_f1",
    "val_f1_weighted": "weighted_f1",
}


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


def resolve_pipeline_seeds() -> tuple[int, ...]:
    """Return seeds from ``TM_RUN_SEEDS``, or :data:`PIPELINE_SEEDS` when unset.

    The env value is a comma-separated list of integers (``42`` or ``42,131``).
    Empty or whitespace-only is treated as unset. A blank entry such as ``42,,131``
    raises ``ValueError``.
    """

    raw = os.environ.get("TM_RUN_SEEDS")
    if raw is None or raw.strip() == "":
        return PIPELINE_SEEDS
    parts = [part.strip() for part in raw.split(",")]
    if any(part == "" for part in parts):
        raise ValueError(
            "TM_RUN_SEEDS must be a comma-separated list of integer seeds; "
            f"got {raw!r}"
        )
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "TM_RUN_SEEDS must be a comma-separated list of integer seeds; "
            f"got {raw!r}"
        ) from exc


def _artifacts_root(root: str | os.PathLike | None) -> Path:
    if root is None:
        return utils_io._SESSION_ARTIFACTS_DIR
    return Path(root)


def seed_is_complete(
    seed: int,
    relative_paths: Sequence[str | os.PathLike],
    *,
    root: str | os.PathLike | None = None,
) -> bool:
    """Return whether every relative file exists under ``seeds/{seed}/``.

    Does not activate that seed. ``root`` defaults to the same session artifact
    root :mod:`utils_io` captured at import.
    """

    seed_dir = _artifacts_root(root) / "seeds" / str(int(seed))
    return all((seed_dir / rel).is_file() for rel in relative_paths)


def _load_json_object(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object")
    return payload


def _metric_row(seed: int, raw: Mapping, key_map: Mapping[str, str], source: Path) -> dict:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source} seed {seed} metric entry must be an object")
    missing = [src for src in key_map if src not in raw]
    if missing:
        raise KeyError(f"{source} seed {seed} missing {', '.join(missing)}")
    row = {"seed": int(seed)}
    for src, dst in key_map.items():
        row[dst] = float(raw[src])
    return row


def _collect_systems_file(
    filename: str,
    system_names: Sequence[str],
    seeds: Sequence[int],
    *,
    root: Path,
    key_map: Mapping[str, str],
) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {name: [] for name in system_names}
    for seed in seeds:
        path = root / "seeds" / str(int(seed)) / "metrics" / filename
        if not path.is_file():
            continue
        payload = _load_json_object(path)
        for name in system_names:
            if name not in payload:
                continue
            runs[name].append(_metric_row(seed, payload[name], key_map, path))
    return runs


def collect_val_runs(
    seeds: Sequence[int] | None = None,
    *,
    root: str | os.PathLike | None = None,
) -> dict[str, list[dict]]:
    """Collect per-seed validation rows for the six paper systems.

    Ensemble systems are read from ``seeds/{seed}/metrics/val_systems.json``.
    Standalone CafeBERT and ViSoBERT are read from ``metrics/{name}.json`` and
    normalized to ``accuracy``, ``macro_f1``, and ``weighted_f1``. A system with
    no rows is omitted. A system missing one seed is kept for the seeds that
    have a row. ``seeds`` defaults to :func:`resolve_pipeline_seeds`.
    """

    base = _artifacts_root(root)
    chosen = resolve_pipeline_seeds() if seeds is None else tuple(int(seed) for seed in seeds)
    runs = _collect_systems_file(
        "val_systems.json",
        ENSEMBLE_VAL_SYSTEMS,
        chosen,
        root=base,
        key_map=_ENSEMBLE_KEY_MAP,
    )
    for name in STANDALONE_VAL_SYSTEMS:
        runs[name] = []
    for seed in chosen:
        metrics_dir = base / "seeds" / str(seed) / "metrics"
        for name in STANDALONE_VAL_SYSTEMS:
            path = metrics_dir / f"{name}.json"
            if not path.is_file():
                continue
            runs[name].append(
                _metric_row(seed, _load_json_object(path), _STANDALONE_KEY_MAP, path)
            )
    return {name: rows for name, rows in runs.items() if rows}


def collect_test_runs(
    seeds: Sequence[int] | None = None,
    *,
    root: str | os.PathLike | None = None,
) -> dict[str, list[dict]]:
    """Collect per-seed test rows for the four ensemble systems.

    Reads ``seeds/{seed}/metrics/test_systems.json``. A system with no rows is
    omitted. A system missing one seed is kept for the seeds that have a row.
    ``seeds`` defaults to :func:`resolve_pipeline_seeds`.
    """

    base = _artifacts_root(root)
    chosen = resolve_pipeline_seeds() if seeds is None else tuple(int(seed) for seed in seeds)
    runs = _collect_systems_file(
        "test_systems.json",
        ENSEMBLE_VAL_SYSTEMS,
        chosen,
        root=base,
        key_map=_ENSEMBLE_KEY_MAP,
    )
    return {name: rows for name, rows in runs.items() if rows}


def write_val_multiseed_summary(
    seeds: Sequence[int] | None = None,
    *,
    root: str | os.PathLike | None = None,
) -> Path:
    """Write ``{session}/metrics/val_multiseed_summary.json`` and return that path.

    The file holds the requested seeds, per-seed runs, and the aggregate from
    :func:`summarize_seed_metrics`. Means are the reported numbers; std and the
    95% CI are kept. The summary stays on the session root, not inside a seed
    directory, even if a seed is currently active.
    """

    base = _artifacts_root(root)
    chosen = resolve_pipeline_seeds() if seeds is None else tuple(int(seed) for seed in seeds)
    collected = collect_val_runs(chosen, root=base)
    summary = {
        "seeds": [int(seed) for seed in chosen],
        "runs": collected,
        "aggregate": {
            name: summarize_seed_metrics(rows, keys=VAL_METRIC_KEYS)
            for name, rows in collected.items()
        },
    }
    out = base / "metrics" / "val_multiseed_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out


def write_test_multiseed_summary(
    seeds: Sequence[int] | None = None,
    *,
    root: str | os.PathLike | None = None,
) -> Path:
    """Write ``{session}/metrics/test_multiseed_summary.json`` and return that path.

    The file holds the requested seeds, per-seed test runs, and the aggregate
    from :func:`summarize_seed_metrics` (mean, sample std, and 95% CI). The
    summary stays on the session root, not inside a seed directory.
    """

    base = _artifacts_root(root)
    chosen = resolve_pipeline_seeds() if seeds is None else tuple(int(seed) for seed in seeds)
    collected = collect_test_runs(chosen, root=base)
    summary = {
        "seeds": [int(seed) for seed in chosen],
        "runs": collected,
        "aggregate": {
            name: summarize_seed_metrics(rows, keys=VAL_METRIC_KEYS)
            for name, rows in collected.items()
        },
    }
    out = base / "metrics" / "test_multiseed_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out
