"""Colab bootstrap: mount Drive, route all ephemeral work to ``/content``.

Persistent artifacts (probs .npy, metrics .json, meta jsonl, final LoRA adapter)
stay inside ``tm_research/ensemble/artifacts/`` which lives on Drive when the
repo is on Drive. Everything transient (HF cache, Trainer checkpoints, LoRA
training checkpoints) goes under ``/content/ensemble_tmp`` so we do NOT pollute
Drive with gigabytes of throwaway files.

Typical first cell on Colab::

    import sys, os
    REPO_ROOT = '/content/drive/MyDrive/thesis/topicmodeling'
    if 'google.colab' in sys.modules:
        from google.colab import drive
        if not os.path.ismount('/content/drive'):
            drive.mount('/content/drive')
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from tm_research.ensemble.colab_setup import setup_colab
    paths = setup_colab(repo_root=REPO_ROOT)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_REPO_ROOT = "/content/drive/MyDrive/thesis/topicmodeling"
DEFAULT_TMP_ROOT = "/content/ensemble_tmp"


@dataclass
class ColabPaths:
    repo_root: Path
    tmp_root: Path
    hf_cache: Path
    bert_work: Path
    lora_work: Path
    artifacts: Path
    persistent_artifacts: Path
    is_colab: bool

    def as_dict(self) -> dict:
        return {
            "repo_root": str(self.repo_root),
            "tmp_root": str(self.tmp_root),
            "hf_cache": str(self.hf_cache),
            "bert_work": str(self.bert_work),
            "lora_work": str(self.lora_work),
            "artifacts": str(self.artifacts),
            "persistent_artifacts": str(self.persistent_artifacts),
            "is_colab": self.is_colab,
        }


def _in_colab() -> bool:
    return "google.colab" in sys.modules


def _maybe_mount_drive() -> None:
    if not _in_colab():
        return
    try:
        from google.colab import drive  # type: ignore

        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
    except ImportError:
        pass


def _maybe_load_hf_token() -> None:
    if os.environ.get("HF_TOKEN"):
        return
    if not _in_colab():
        return
    try:
        from google.colab import userdata  # type: ignore

        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
    except Exception:
        pass


def setup_colab(
    repo_root: str | os.PathLike = DEFAULT_REPO_ROOT,
    tmp_root: str | os.PathLike = DEFAULT_TMP_ROOT,
    require_repo: bool = True,
    load_hf_token: bool = True,
    pull_artifacts: bool = True,
) -> ColabPaths:
    """Prepare Colab so Drive holds artifacts and ``/content`` holds scratch.

    Creates ``{tmp_root}/{hf_cache,bert_work,lora_work}`` and sets
    ``HF_HOME`` / ``HF_DATASETS_CACHE`` / ``TRANSFORMERS_CACHE`` env vars to
    point at ``hf_cache`` so model downloads also stay on ephemeral disk.

    Mounts Drive if running in Colab and Drive is not mounted.
    ``chdir``s into ``repo_root`` and prepends it to ``sys.path`` so
    ``from tm_research.ensemble...`` imports work.

    Returns the resolved paths. Safe to call outside Colab - it becomes a
    no-op aside from optionally putting the local repo root on ``sys.path``.
    """

    is_colab = _in_colab()
    repo_path = Path(repo_root).resolve() if not is_colab else Path(repo_root)
    tmp_path = Path(tmp_root)

    if is_colab:
        _maybe_mount_drive()
        if require_repo and not repo_path.exists():
            raise FileNotFoundError(
                f"Repo not found at {repo_path}. Upload the repo to Drive or "
                "pass a different repo_root= to setup_colab()."
            )
        tmp_path.mkdir(parents=True, exist_ok=True)

    hf_cache = tmp_path / "hf_cache"
    bert_work = tmp_path / "bert_work"
    lora_work = tmp_path / "lora_work"
    local_artifacts = tmp_path / "artifacts"
    persistent_artifacts = repo_path / "tm_research" / "ensemble" / "artifacts"

    for p in (hf_cache, bert_work, lora_work):
        if is_colab or tmp_path.exists():
            p.mkdir(parents=True, exist_ok=True)

    if is_colab:
        local_artifacts.mkdir(parents=True, exist_ok=True)
        os.environ["TM_ENSEMBLE_ARTIFACTS_DIR"] = str(local_artifacts)
        os.environ["TM_ENSEMBLE_PERSISTENT_ARTIFACTS_DIR"] = str(persistent_artifacts)
        artifacts_in_use = local_artifacts
    else:
        artifacts_in_use = persistent_artifacts

    if is_colab:
        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["HF_DATASETS_CACHE"] = str(hf_cache)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
        os.environ.setdefault("WANDB_MODE", "disabled")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        try:
            os.chdir(repo_path)
        except FileNotFoundError:
            pass
    else:
        if repo_path.exists() and str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

    if load_hf_token:
        _maybe_load_hf_token()

    if is_colab and pull_artifacts:
        try:
            from tm_research.ensemble.utils_io import pull_artifacts_from_persistent

            pull_artifacts_from_persistent(verbose=True)
        except Exception as exc:
            print(f"[colab_setup] artifact pull skipped ({type(exc).__name__}: {exc})")

    paths = ColabPaths(
        repo_root=repo_path,
        tmp_root=tmp_path,
        hf_cache=hf_cache,
        bert_work=bert_work,
        lora_work=lora_work,
        artifacts=artifacts_in_use,
        persistent_artifacts=persistent_artifacts,
        is_colab=is_colab,
    )
    print(f"[colab_setup] is_colab={is_colab}")
    print(f"[colab_setup] repo_root={repo_path}")
    print(f"[colab_setup] tmp_root={tmp_path} (hf_cache/bert_work/lora_work)")
    print(f"[colab_setup] artifacts={artifacts_in_use}")
    print(f"[colab_setup] persistent_artifacts={persistent_artifacts}")
    if is_colab:
        hf = "set" if os.environ.get("HF_TOKEN") else "NOT set"
        print(f"[colab_setup] HF_TOKEN: {hf}")
    return paths
