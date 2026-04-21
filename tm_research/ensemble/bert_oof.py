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
    n_folds: int = 5
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
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

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
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=fp16,
        bf16=bf16,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )
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
    oof_acc = float((oof_probs.argmax(axis=1) == train_labels).mean())

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
    val_acc = float((val_probs.argmax(axis=1) == val_labels).mean())

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
    test_acc = float((test_probs.argmax(axis=1) == test_labels).mean())

    elapsed = time.time() - t0
    metrics = {
        "model_name": cfg.model_name,
        "output_name": cfg.output_name,
        "seed": cfg.seed,
        "n_folds": cfg.n_folds,
        "fold_val_accuracies": fold_accs,
        "oof_accuracy": oof_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "num_train": int(len(train_texts)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "elapsed_seconds": elapsed,
    }
    save_metrics(cfg.output_name, metrics)
    print(f"[{cfg.output_name}] done in {elapsed/60:.1f} min — oof={oof_acc:.4f} val={val_acc:.4f} test={test_acc:.4f}")
    return metrics
