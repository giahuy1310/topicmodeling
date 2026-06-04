"""Colab bootstrap for VSFC_ensemble (isolated artifacts and scratch dirs)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_REPO_ROOT = "/content/drive/MyDrive/thesis/topicmodeling/tm_research"
DEFAULT_TMP_ROOT = "/content/vsfc_ensemble_tmp"


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


def _resolve_project_paths(repo_path: Path) -> tuple[Path, Path]:
    if repo_path.name == "tm_research":
        import_root = repo_path.parent
        persistent_artifacts = repo_path / "VSFC_ensemble" / "artifacts"
    else:
        import_root = repo_path
        persistent_artifacts = repo_path / "tm_research" / "VSFC_ensemble" / "artifacts"
    return import_root, persistent_artifacts


def setup_colab(
    repo_root: str | os.PathLike = DEFAULT_REPO_ROOT,
    tmp_root: str | os.PathLike = DEFAULT_TMP_ROOT,
    require_repo: bool = True,
    load_hf_token: bool = True,
    pull_artifacts: bool = True,
) -> ColabPaths:
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
    import_root, persistent_artifacts = _resolve_project_paths(repo_path)

    for p in (hf_cache, bert_work, lora_work):
        if is_colab or tmp_path.exists():
            p.mkdir(parents=True, exist_ok=True)

    if is_colab:
        local_artifacts.mkdir(parents=True, exist_ok=True)
        os.environ["TM_VSFC_ENSEMBLE_ARTIFACTS_DIR"] = str(local_artifacts)
        os.environ["TM_VSFC_ENSEMBLE_PERSISTENT_ARTIFACTS_DIR"] = str(persistent_artifacts)
        artifacts_in_use = local_artifacts
    else:
        artifacts_in_use = persistent_artifacts

    if is_colab:
        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["HF_DATASETS_CACHE"] = str(hf_cache)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
        os.environ.setdefault("WANDB_MODE", "disabled")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        if str(import_root) not in sys.path:
            sys.path.append(str(import_root))
        try:
            os.chdir(str(tmp_path))
        except FileNotFoundError:
            pass
    else:
        if repo_path.exists() and str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

    if load_hf_token:
        _maybe_load_hf_token()

    if is_colab and pull_artifacts:
        try:
            from tm_research.VSFC_ensemble.utils_io import pull_artifacts_from_persistent

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
    print(f"[colab_setup] tmp_root={tmp_path}")
    print(f"[colab_setup] artifacts={artifacts_in_use}")
    print(f"[colab_setup] persistent_artifacts={persistent_artifacts}")
    if is_colab:
        hf = "set" if os.environ.get("HF_TOKEN") else "NOT set"
        print(f"[colab_setup] HF_TOKEN: {hf}")
    return paths
