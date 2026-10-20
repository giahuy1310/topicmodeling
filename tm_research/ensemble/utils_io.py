"""Shared I/O helpers for the ensemble pipeline.

Loads the train/val/test CSVs, normalizes column names to ``text`` and ``label``,
and builds deterministic ``label2id`` / ``id2label`` mappings from the training
labels (sorted alphabetically, matching the convention used by the existing
PhoBERT/CafeBERT/ViBERT notebooks).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

PERSISTENT_ARTIFACTS_DIR = Path(
    os.environ.get(
        "TM_ENSEMBLE_PERSISTENT_ARTIFACTS_DIR",
        str(Path(__file__).resolve().parent / "artifacts"),
    )
)

ARTIFACTS_DIR = Path(
    os.environ.get("TM_ENSEMBLE_ARTIFACTS_DIR", str(PERSISTENT_ARTIFACTS_DIR))
)
# Session root captured at import. Seed activation rebinds ARTIFACTS_DIR, but
# push/pull always copy this directory so one seed folder cannot flatten onto Drive.
_SESSION_ARTIFACTS_DIR = ARTIFACTS_DIR
PROBS_DIR = ARTIFACTS_DIR / "probs"
META_JSONL_DIR = ARTIFACTS_DIR / "meta_jsonl"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
LORA_DIR = ARTIFACTS_DIR / "lora_adapter"

TRAIN_FILE = "train_1500_final.csv"
VAL_FILE = "val_final.csv"
TEST_FILE = "test_final.csv"

TEXT_CANDIDATES = ("Sentence_clean", "Sentence", "text")
LABEL_CANDIDATES = ("Emotion", "label")


@dataclass
class LabelMap:
    label2id: Dict[str, int]
    id2label: Dict[int, str]

    @property
    def num_classes(self) -> int:
        return len(self.label2id)

    @property
    def class_names(self) -> List[str]:
        return [self.id2label[i] for i in range(self.num_classes)]


def _ensure_dirs() -> None:
    for d in (ARTIFACTS_DIR, PROBS_DIR, META_JSONL_DIR, METRICS_DIR, LORA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _rebind_artifact_dirs(root: Path) -> None:
    global ARTIFACTS_DIR, PROBS_DIR, META_JSONL_DIR, METRICS_DIR, LORA_DIR
    ARTIFACTS_DIR = root
    PROBS_DIR = root / "probs"
    META_JSONL_DIR = root / "meta_jsonl"
    METRICS_DIR = root / "metrics"
    LORA_DIR = root / "lora_adapter"


def activate_seed_artifacts(seed: int, *, root: str | os.PathLike | None = None) -> Path:
    """Rebind artifact globals to ``{session}/seeds/{seed}/`` and create them.

    ``root`` overrides the session root for tests. The default is
    ``_SESSION_ARTIFACTS_DIR``, the artifact root captured at import.
    Returns the seed directory.
    """

    base = Path(root) if root is not None else _SESSION_ARTIFACTS_DIR
    seed_dir = base / "seeds" / str(int(seed))
    _rebind_artifact_dirs(seed_dir)
    _ensure_dirs()
    return seed_dir


def reset_artifacts_root() -> Path:
    """Restore artifact globals to the session root captured at import.

    Returns that session root. Does not delete seed directories.
    """

    _rebind_artifact_dirs(_SESSION_ARTIFACTS_DIR)
    return _SESSION_ARTIFACTS_DIR


def _pick_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _pick_column(df, TEXT_CANDIDATES)
    label_col = _pick_column(df, LABEL_CANDIDATES)
    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str).str.strip(),
            "label": df[label_col].astype(str).str.strip(),
        }
    )
    out = out[(out["text"] != "") & (out["label"] != "")].reset_index(drop=True)
    return out


def load_splits(
    data_dir: str | os.PathLike | None = None,
    train_file: str = TRAIN_FILE,
    val_file: str = VAL_FILE,
    test_file: str = TEST_FILE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelMap]:
    """Load train/val/test as ``(text, label)`` DataFrames plus the label map.

    The label map is built from sorted unique labels in the training set so the
    indexing is stable across runs.
    """

    base = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    train_df = _normalize(pd.read_csv(base / train_file))
    val_df = _normalize(pd.read_csv(base / val_file))
    test_df = _normalize(pd.read_csv(base / test_file))

    labels = sorted(train_df["label"].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    val_df = val_df[val_df["label"].isin(label2id)].reset_index(drop=True)
    test_df = test_df[test_df["label"].isin(label2id)].reset_index(drop=True)

    _ensure_dirs()
    with open(ARTIFACTS_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return train_df, val_df, test_df, LabelMap(label2id=label2id, id2label=id2label)


def load_label_map() -> LabelMap:
    """Reload the label map saved by :func:`load_splits`."""

    path = ARTIFACTS_DIR / "label_map.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    label2id = {k: int(v) for k, v in raw["label2id"].items()}
    id2label = {int(k): v for k, v in raw["id2label"].items()}
    return LabelMap(label2id=label2id, id2label=id2label)


def probs_path(model_name: str, split: str) -> Path:
    """Return ``artifacts/probs/{model_name}_{split}.npy``."""

    assert split in {"oof", "val", "test"}, split
    return PROBS_DIR / f"{model_name}_{split}.npy"


def save_probs(model_name: str, split: str, probs: np.ndarray) -> Path:
    _ensure_dirs()
    p = probs_path(model_name, split)
    np.save(p, probs.astype(np.float32))
    return p


def load_probs(model_name: str, split: str) -> np.ndarray:
    return np.load(probs_path(model_name, split))


def save_metrics(model_name: str, payload: dict) -> Path:
    _ensure_dirs()
    p = METRICS_DIR / f"{model_name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def load_metrics(model_name: str) -> dict:
    with open(METRICS_DIR / f"{model_name}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _copy_tree(src: Path, dst: Path, *, newer_only: bool, verbose: bool) -> int:
    if not src.exists():
        if verbose:
            print(f"[utils_io] skip: {src} does not exist")
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for srcp in src.rglob("*"):
        if srcp.is_dir():
            continue
        rel = srcp.relative_to(src)
        dstp = dst / rel
        if newer_only and dstp.exists():
            try:
                if dstp.stat().st_mtime >= srcp.stat().st_mtime and dstp.stat().st_size == srcp.stat().st_size:
                    continue
            except OSError:
                pass
        dstp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(srcp, dstp)
        n += 1
    return n


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return str(a) == str(b)


def push_artifacts_to_persistent(verbose: bool = True) -> Optional[Path]:
    """Copy the session artifact root → ``PERSISTENT_ARTIFACTS_DIR``.

    Always copies ``_SESSION_ARTIFACTS_DIR``, including while a seed directory
    is active, so one seed folder is not flattened onto the Drive artifact root.
    When no seed is active this matches copying ``ARTIFACTS_DIR``. No-op if the
    session root and the persistent root resolve to the same directory.

    Run this at the end of a Colab notebook so results on local scratch
    (``/content/...``) are persisted to Drive.
    """

    src, dst = _SESSION_ARTIFACTS_DIR, PERSISTENT_ARTIFACTS_DIR
    if _same_path(src, dst):
        if verbose:
            print(f"[utils_io] push skipped: session artifacts == PERSISTENT_ARTIFACTS_DIR ({src})")
        return None
    n = _copy_tree(src, dst, newer_only=False, verbose=verbose)
    if verbose:
        print(f"[utils_io] pushed {n} files: {src} -> {dst}")
    return dst


def pull_artifacts_from_persistent(verbose: bool = True) -> Optional[Path]:
    """Copy ``PERSISTENT_ARTIFACTS_DIR`` → session artifact root (newer files only).

    Always writes to ``_SESSION_ARTIFACTS_DIR``, including while a seed directory
    is active. When no seed is active this matches copying into ``ARTIFACTS_DIR``.
    Called by ``colab_setup.setup_colab`` at session start so a fresh Colab
    VM can see artifacts produced by earlier runs (e.g. notebook 05 reading
    OOF probs written by 01..04 last week).
    """

    src, dst = PERSISTENT_ARTIFACTS_DIR, _SESSION_ARTIFACTS_DIR
    if _same_path(src, dst):
        if verbose:
            print(f"[utils_io] pull skipped: session artifacts == PERSISTENT_ARTIFACTS_DIR ({dst})")
        return None
    n = _copy_tree(src, dst, newer_only=True, verbose=verbose)
    if verbose:
        print(f"[utils_io] pulled {n} files: {src} -> {dst}")
    return dst
