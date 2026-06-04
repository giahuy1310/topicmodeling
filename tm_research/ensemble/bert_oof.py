"""Shared K-fold OOF + full-train refit routine for HuggingFace BERT models.

The three BERT base-model notebooks (PhoBERT, CafeBERT, ViBERT) all call
``run_bert_oof`` with a different ``model_name`` / ``output_name`` / ``seed``;
everything else is identical.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score

from .utils_io import (
    LabelMap,
    METRICS_DIR,
    save_metrics,
    save_probs,
)
from .utils_stacking import kfold_oof_probs


@dataclass
class BertOOFConfig:
    model_name: str
    output_name: str
    seed: int = 42
    n_folds: int = 3
    num_epochs: int = 6
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    train_batch_size: int = 32
    eval_batch_size: int = 64
    grad_accum_steps: int = 1
    max_length: int = 256
    work_dir: str = "./_bert_oof_work"
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    early_stopping_patience: int = 2
    extra_tokenizer_kwargs: Dict[str, object] = field(default_factory=dict)
    extra_model_kwargs: Dict[str, object] = field(default_factory=dict)
    # Scheduler / optimizer tuning.
    lr_scheduler_type: str = "cosine"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    metric_for_best_model: str = "f1_macro"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _device_precision_flags(cfg: BertOOFConfig):
    import torch

    fp16 = cfg.fp16
    bf16 = cfg.bf16
    if fp16 is None and bf16 is None:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 8:
                bf16 = True
                fp16 = False
            else:
                fp16 = True
                bf16 = False
        else:
            fp16 = False
            bf16 = False
    return bool(fp16), bool(bf16)


def _build_trainer(
    cfg: BertOOFConfig,
    label_map: LabelMap,
    train_texts,
    train_labels,
    eval_texts,
    eval_labels,
    seed: int,
    output_dir: str,
):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **cfg.extra_tokenizer_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=label_map.num_classes,
        id2label=label_map.id2label,
        label2id=label_map.label2id,
        **cfg.extra_model_kwargs,
    )

    def to_ds(texts, labels):
        df = pd.DataFrame({"text": list(texts), "label": list(labels)})
        return Dataset.from_pandas(df, preserve_index=False)

    train_ds = to_ds(train_texts, train_labels)
    eval_ds = to_ds(eval_texts, eval_labels)

    def tok(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=cfg.max_length,
        )

    train_ds = train_ds.map(tok, batched=True)
    eval_ds = eval_ds.map(tok, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        return {
            "accuracy": acc,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_macro": f1_macro,
            "precision_weighted": p_w,
            "recall_weighted": r_w,
            "f1_weighted": f1_w,
        }

    fp16, bf16 = _device_precision_flags(cfg)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=fp16,
        bf16=bf16,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    import inspect

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    return trainer, tokenizer, eval_ds


def _predict_probs(trainer, dataset) -> np.ndarray:
    import torch.nn.functional as F
    import torch

    pred = trainer.predict(dataset)
    logits = pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = F.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=-1).numpy()
    return probs


def run_bert_oof(
    cfg: BertOOFConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_map: LabelMap,
) -> Dict[str, object]:
    """Run K-fold OOF on ``train_df`` then refit on full train and predict val/test.

    Saves:
      ``artifacts/probs/{output_name}_oof.npy``,
      ``artifacts/probs/{output_name}_val.npy``,
      ``artifacts/probs/{output_name}_test.npy``,
      ``artifacts/metrics/{output_name}.json``.

    Returns the metrics dict.
    """

    _seed_everything(cfg.seed)
    work_root = Path(cfg.work_dir).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].map(label_map.label2id).to_numpy()

    def _fold_fit_predict(x_tr, y_tr, x_va) -> np.ndarray:
        fold_dir = work_root / f"{cfg.output_name}_fold"
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
        trainer, _tok, eval_ds = _build_trainer(
            cfg,
            label_map,
            list(x_tr),
            list(y_tr),
            list(x_va),
            [0] * len(x_va),
            seed=cfg.seed,
            output_dir=str(fold_dir),
        )
        trainer.train()
        probs = _predict_probs(trainer, eval_ds)
        del trainer
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
        return probs

    print(f"[{cfg.output_name}] running {cfg.n_folds}-fold OOF on N={len(train_texts)}")
    t0 = time.time()
    oof_probs, fold_accs = kfold_oof_probs(
        _fold_fit_predict,
        train_texts,
        train_labels,
        num_classes=label_map.num_classes,
        n_splits=cfg.n_folds,
        seed=cfg.seed,
    )
    save_probs(cfg.output_name, "oof", oof_probs)
    oof_pred = oof_probs.argmax(axis=1)
    oof_acc = float(accuracy_score(train_labels, oof_pred))
    oof_recall = float(recall_score(train_labels, oof_pred, average="macro", zero_division=0))
    oof_f1 = float(f1_score(train_labels, oof_pred, average="macro", zero_division=0))
    oof_f1_weighted = float(f1_score(train_labels, oof_pred, average="weighted", zero_division=0))

    print(f"[{cfg.output_name}] refitting on full train and predicting val/test")
    val_labels = val_df["label"].map(label_map.label2id).to_numpy()
    test_labels = test_df["label"].map(label_map.label2id).to_numpy()
    full_dir = work_root / f"{cfg.output_name}_full"
    if full_dir.exists():
        shutil.rmtree(full_dir)
    trainer, _tok, val_ds = _build_trainer(
        cfg,
        label_map,
        train_texts,
        train_labels,
        val_df["text"].tolist(),
        val_labels,
        seed=cfg.seed,
        output_dir=str(full_dir),
    )
    trainer.train()
    val_probs = _predict_probs(trainer, val_ds)
    save_probs(cfg.output_name, "val", val_probs)
    val_pred = val_probs.argmax(axis=1)
    val_acc = float(accuracy_score(val_labels, val_pred))
    val_recall = float(recall_score(val_labels, val_pred, average="macro", zero_division=0))
    val_f1 = float(f1_score(val_labels, val_pred, average="macro", zero_division=0))
    val_f1_weighted = float(f1_score(val_labels, val_pred, average="weighted", zero_division=0))

    from datasets import Dataset

    test_ds_only = Dataset.from_pandas(
        pd.DataFrame({"text": test_df["text"].tolist(), "label": test_labels.tolist()}),
        preserve_index=False,
    )
    tokenizer = _tok

    def _tok_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=cfg.max_length,
        )

    test_ds_only = test_ds_only.map(_tok_fn, batched=True)
    test_probs = _predict_probs(trainer, test_ds_only)
    save_probs(cfg.output_name, "test", test_probs)
    test_pred = test_probs.argmax(axis=1)
    test_acc = float(accuracy_score(test_labels, test_pred))
    test_recall = float(recall_score(test_labels, test_pred, average="macro", zero_division=0))
    test_f1 = float(f1_score(test_labels, test_pred, average="macro", zero_division=0))
    test_f1_weighted = float(f1_score(test_labels, test_pred, average="weighted", zero_division=0))

    elapsed = time.time() - t0
    metrics = {
        "model_name": cfg.model_name,
        "output_name": cfg.output_name,
        "seed": cfg.seed,
        "n_folds": cfg.n_folds,
        "fold_val_accuracies": fold_accs,
        "oof_accuracy": oof_acc,
        "oof_recall": oof_recall,
        "oof_f1": oof_f1,
        "oof_f1_weighted": oof_f1_weighted,
        "val_accuracy": val_acc,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_f1_weighted": val_f1_weighted,
        "test_accuracy": test_acc,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_f1_weighted": test_f1_weighted,
        "num_train": int(len(train_texts)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "elapsed_seconds": elapsed,
        "tuning": {
            "learning_rate": cfg.learning_rate,
            "lr_scheduler_type": cfg.lr_scheduler_type,
            "warmup_ratio": cfg.warmup_ratio,
            "weight_decay": cfg.weight_decay,
            "num_epochs": cfg.num_epochs,
            "train_batch_size": cfg.train_batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "max_length": cfg.max_length,
            "early_stopping_patience": cfg.early_stopping_patience,
            "max_grad_norm": cfg.max_grad_norm,
        },
    }
    save_metrics(cfg.output_name, metrics)
    print(f"[{cfg.output_name}] done in {elapsed/60:.1f} min — oof={oof_acc:.4f} val={val_acc:.4f} test={test_acc:.4f}")
    return metrics
