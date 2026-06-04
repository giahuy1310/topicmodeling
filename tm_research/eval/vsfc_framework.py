"""Shared helpers for the UIT-VSFC framework evaluation notebook.

Supports two evaluation strategies:
  - Zero-shot: all components trained on UIT-VSMEC, applied directly to VSFC.
  - Light adapt: BERT backbones frozen; sklearn retrained on VSFC (3-class);
    ensemble weights recomputed on VSFC val; LoRA meta continued on VSFC.

Public API
----------
collect_bert_probs(model_dirs, texts, cache_dir, split_tag)
build_sklearn_vsmec(train_df, label2id)
build_sklearn_vsfc(train_df)
weighted_sentiment_zeroshot(bert_probs_7, sklearn_probs_7, class_names, weights)
weighted_sentiment_adapted(bert_probs_7, sklearn_probs_3, class_names, weights)
compute_vsfc_weights(bert_probs_7_val, sklearn_probs_3_val, class_names, gold_val)
build_vsfc_meta_jsonl(texts, bert_probs_7, sklearn_probs_3, weights, labels, jsonl_path)
run_gemma_zeroshot(texts, base_probs_7, weights, vsmec_label_map, lora_dir,
                   system_prompt, device, batch_size, max_new_tokens)
run_gemma_adapted(texts, base_probs_3, weights_3, vsfc_label_map, lora_dir,
                  system_prompt, device, batch_size, max_new_tokens)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# BERT prob collection (7-class, cached)
# ---------------------------------------------------------------------------

def collect_bert_probs(
    model_dirs: Dict[str, Optional[str]],
    texts: List[str],
    cache_dir: Path,
    split_tag: str,
    device: Optional[str] = None,
    force_recompute: bool = False,
) -> Dict[str, np.ndarray]:
    """Run each BERT emotion checkpoint on *texts* and cache results.

    Parameters
    ----------
    model_dirs : mapping from model key to checkpoint directory (or None to skip).
    texts : raw Vietnamese text strings.
    cache_dir : directory where ``vsfc_{key}_{split_tag}_probs.npy`` are stored.
    split_tag : e.g. ``"test"`` or ``"val"`` (used in the cache filename).
    device : torch device string; auto-detects if None.
    force_recompute : ignore cached files and re-run GPU inference.

    Returns
    -------
    dict mapping model key → (N, 7) float32 probability array.
    """
    import torch
    from tm_research.eval.bert_emotion_probs import load_emotion_classifier, predict_probs

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, np.ndarray] = {}
    for key, model_dir in model_dirs.items():
        if model_dir is None or not Path(model_dir).exists():
            print(f"  [{key}] checkpoint not found — skipped.")
            continue

        cache_path = cache_dir / f"vsfc_{key}_{split_tag}_probs.npy"
        if cache_path.exists() and not force_recompute:
            result[key] = np.load(cache_path)
            print(f"  [{key}] loaded from cache {cache_path}")
            continue

        print(f"  [{key}] running inference …")
        model, tokenizer, _ = load_emotion_classifier(model_dir)
        probs = predict_probs(model, tokenizer, texts, device=device)
        np.save(cache_path, probs)
        print(f"  [{key}] cached → {cache_path}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        result[key] = probs

    return result


def get_bert_class_names(model_dirs: Dict[str, Optional[str]], key: str) -> Optional[List[str]]:
    """Read class_names from label_mappings.json without loading model weights."""
    model_dir = model_dirs.get(key)
    if model_dir is None or not Path(model_dir).exists():
        return None
    lm_path = Path(model_dir) / "label_mappings.json"
    with open(lm_path, encoding="utf-8") as f:
        lm = json.load(f)
    id2label = {int(k): v for k, v in lm["id2label"].items()}
    return [id2label[i] for i in range(len(id2label))]


# ---------------------------------------------------------------------------
# Sklearn base builders
# ---------------------------------------------------------------------------

def _make_logreg():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, max_features=50_000, ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                     multi_class="multinomial", class_weight="balanced",
                                     random_state=42)),
    ])


def _make_svc():
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, max_features=50_000, ngram_range=(1, 2))),
        ("clf",   CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42))),
    ])


def build_sklearn_vsmec(
    train_df,
    label2id: Dict[str, int],
) -> Tuple[object, object, List[str]]:
    """Fit TF-IDF + LogReg and TF-IDF + CalibratedSVC on UIT-VSMEC train (7-class).

    Returns (logreg_pipe, svc_pipe, class_names_ordered_by_id).
    """
    import numpy as np
    x_tr = train_df["text"].tolist()
    y_tr = train_df["label"].map(label2id).to_numpy()
    class_names = [k for k, _ in sorted(label2id.items(), key=lambda kv: kv[1])]

    logreg = _make_logreg()
    svc    = _make_svc()
    logreg.fit(x_tr, y_tr)
    svc.fit(x_tr, y_tr)
    print(f"  VSMEC sklearn fitted on {len(x_tr)} examples, {len(class_names)} classes.")
    return logreg, svc, class_names


def build_sklearn_vsfc(
    train_texts: List[str],
    train_labels: List[str],
) -> Tuple[object, object]:
    """Fit TF-IDF + LogReg and CalibratedSVC on VSFC train (3-class sentiment).

    Labels are the string polarity labels: "negative", "neutral", "positive".
    Returns (logreg_pipe, svc_pipe); predict_proba gives (N, 3) arrays
    in sorted-label order ["negative", "neutral", "positive"].
    """
    logreg = _make_logreg()
    svc    = _make_svc()
    logreg.fit(train_texts, train_labels)
    svc.fit(train_texts, train_labels)
    print(f"  VSFC sklearn fitted on {len(train_texts)} examples.")
    return logreg, svc


# ---------------------------------------------------------------------------
# Weighted ensemble (zero-shot path — all probs are 7-class)
# ---------------------------------------------------------------------------

def weighted_sentiment_zeroshot(
    bert_probs_7: Dict[str, np.ndarray],
    sklearn_probs_7: Dict[str, np.ndarray],
    class_names_7: List[str],
    weights: Dict[str, float],
) -> np.ndarray:
    """Weighted average in 7-class space, then collapse to 3-class.

    Parameters
    ----------
    bert_probs_7 : ``{model_key: (N, 7)}`` arrays for BERT models.
    sklearn_probs_7 : ``{model_key: (N, 7)}`` arrays for sklearn models.
    class_names_7 : ordered 7-class emotion names (same for all models).
    weights : per-model weights from ``weights.json``.

    Returns
    -------
    (N, 3) collapsed probability array [negative, neutral, positive].
    """
    from tm_research.ensemble.utils_stacking import weighted_average
    from tm_research.eval.sentiment_collapse import collapse_batch

    all_probs = {**bert_probs_7, **sklearn_probs_7}
    usable = {k: v for k, v in all_probs.items() if k in weights}
    if not usable:
        raise ValueError("No models found in both all_probs and weights.")
    avg_7 = weighted_average(usable, weights)
    return collapse_batch(avg_7, class_names_7)


# ---------------------------------------------------------------------------
# Weighted ensemble (light-adapt path — BERT 7-class, sklearn 3-class)
# ---------------------------------------------------------------------------

def weighted_sentiment_adapted(
    bert_probs_7: Dict[str, np.ndarray],
    sklearn_probs_3: Dict[str, np.ndarray],
    class_names_7: List[str],
    weights: Dict[str, float],
) -> np.ndarray:
    """3-class weighted average: collapse BERT 7→3 first, then average with sklearn 3-class.

    Parameters
    ----------
    bert_probs_7 : ``{model_key: (N, 7)}`` for BERT models.
    sklearn_probs_3 : ``{model_key: (N, 3)}`` for VSFC-adapted sklearn models.
        Column order must match POLARITY_CLASSES = ["negative", "neutral", "positive"].
    class_names_7 : 7-class emotion names for collapse.
    weights : per-model weights recomputed on VSFC val.

    Returns
    -------
    (N, 3) final sentiment probability array.
    """
    from tm_research.eval.sentiment_collapse import collapse_batch

    all_probs_3: Dict[str, np.ndarray] = {}
    for key, p7 in bert_probs_7.items():
        if key in weights:
            all_probs_3[key] = collapse_batch(p7, class_names_7)
    for key, p3 in sklearn_probs_3.items():
        if key in weights:
            all_probs_3[key] = np.asarray(p3, dtype=np.float32)

    keys = [k for k in all_probs_3 if k in weights]
    if not keys:
        raise ValueError("No overlapping models in all_probs_3 and weights.")

    P = np.stack([all_probs_3[k] for k in keys], axis=0)  # (M, N, 3)
    w = np.array([weights[k] for k in keys], dtype=np.float32)
    w = w / w.sum()
    return np.tensordot(w, P, axes=([0], [0]))  # (N, 3)


# ---------------------------------------------------------------------------
# Weight recomputation on VSFC validation
# ---------------------------------------------------------------------------

def compute_vsfc_weights(
    bert_probs_7_val: Dict[str, np.ndarray],
    sklearn_probs_3_val: Dict[str, np.ndarray],
    class_names_7: List[str],
    gold_val: List[str],
) -> Dict[str, float]:
    """Compute normalized per-model 3-class accuracy on VSFC validation.

    BERT models are collapsed 7→3 before accuracy computation.
    Returns ``{model_key: weight}`` (accuracy-normalized).
    """
    from tm_research.eval.sentiment_collapse import collapse_batch, probs_to_pred_labels
    from tm_research.ensemble.utils_stacking import compute_weights

    gold = np.asarray(gold_val)
    acc_dict: Dict[str, float] = {}

    for key, p7 in bert_probs_7_val.items():
        p3 = collapse_batch(p7, class_names_7)
        preds = np.asarray(probs_to_pred_labels(p3))
        acc_dict[key] = float((preds == gold).mean())

    for key, p3 in sklearn_probs_3_val.items():
        from tm_research.eval.sentiment_collapse import POLARITY_CLASSES
        class_order = POLARITY_CLASSES  # ["negative", "neutral", "positive"]
        preds = np.array([class_order[i] for i in p3.argmax(axis=1)])
        acc_dict[key] = float((preds == gold).mean())

    print("  VSFC val accuracy per model:")
    for k, a in sorted(acc_dict.items(), key=lambda kv: -kv[1]):
        print(f"    {k:12s}: {a:.4f}")

    return compute_weights(acc_dict)


# ---------------------------------------------------------------------------
# VSFC meta JSONL builder (3-class completions)
# ---------------------------------------------------------------------------

def build_vsfc_meta_jsonl(
    texts: List[str],
    bert_probs_3: Dict[str, np.ndarray],
    sklearn_probs_3: Dict[str, np.ndarray],
    weights: Dict[str, float],
    labels: Optional[List[str]],
    jsonl_path: Path,
    system_prompt: str,
) -> Path:
    """Build structured-token prompt JSONL for VSFC LoRA continuation.

    Uses a 3-class prompt schema (negative / neutral / positive).

    Parameters
    ----------
    texts : input texts.
    bert_probs_3 : BERT probs already collapsed to 3-class ``{key: (N,3)}``.
    sklearn_probs_3 : sklearn 3-class probs (VSFC-fitted) ``{key: (N,3)}``.
    weights : VSFC val-tuned weights.
    labels : gold sentiment strings; None produces inference-style rows (no completion).
    jsonl_path : destination file.
    system_prompt : prepended to each prompt string.
    """
    from tm_research.eval.vsfc_format_prompt import format_prompt_3class

    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    all_probs_3: Dict[str, np.ndarray] = {}
    for k, v in bert_probs_3.items():
        if k in weights:
            all_probs_3[k] = np.asarray(v, dtype=np.float32)
    for k, v in sklearn_probs_3.items():
        if k in weights:
            all_probs_3[k] = np.asarray(v, dtype=np.float32)

    rows: List[dict] = []
    N = len(texts)
    for i in range(N):
        row_probs = {k: all_probs_3[k][i] for k in all_probs_3}
        fp = format_prompt_3class(
            text=texts[i],
            probs_per_model=row_probs,
            weights=weights,
            label=labels[i] if labels is not None else None,
        )
        row: dict = {
            "prompt": f"{system_prompt}\n\n{fp['prompt']}",
            "completion": fp["completion"],
        }
        if labels is not None:
            row["gold"] = labels[i]
        rows.append(row)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Wrote {N} rows → {jsonl_path}")
    return jsonl_path


# ---------------------------------------------------------------------------
# Gemma inference — zero-shot (7-class output → polarity collapse)
# ---------------------------------------------------------------------------

def run_gemma_zeroshot(
    texts: List[str],
    base_probs_7: Dict[str, np.ndarray],
    weights: Dict[str, float],
    vsmec_label_map,
    lora_dir: Path,
    system_prompt: str,
    gemma_base_model: str = _GEMMA_DEFAULT,
    device: Optional[str] = None,
    batch_size: int = 4,
    max_new_tokens: int = 12,
    max_length: int = 640,
) -> List[Optional[str]]:
    """Run VSMEC-adapted Gemma on VSFC texts; map output emotion → polarity.

    Returns list of predicted polarity strings ("negative"/"neutral"/"positive"),
    with None for unparseable completions.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from tm_research.ensemble.utils_stacking import format_prompt, parse_label_from_completion
    from tm_research.eval.sentiment_collapse import EMOTION_TO_POLARITY

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(str(lora_dir))
    base_model = AutoModelForCausalLM.from_pretrained(
        gemma_base_model, device_map="auto", torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, str(lora_dir))
    model.eval()

    usable_weights = {k: v for k, v in weights.items() if k in base_probs_7}
    preds: List[Optional[str]] = []
    N = len(texts)

    for i in range(0, N, batch_size):
        batch_texts = texts[i: i + batch_size]
        batch_size_cur = len(batch_texts)
        prompts = []
        for j in range(batch_size_cur):
            row_probs = {m: base_probs_7[m][i + j] for m in usable_weights if m in base_probs_7}
            fp = format_prompt(batch_texts[j], row_probs, usable_weights, vsmec_label_map, label=None)
            prompts.append(f"{system_prompt}\n\n{fp['prompt']}")

        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.eos_token_id, do_sample=False)
        inp_len = enc["input_ids"].shape[1]
        for seq in out:
            new_text = tok.decode(seq[inp_len:], skip_special_tokens=True)
            emotion = parse_label_from_completion(new_text, vsmec_label_map)
            preds.append(EMOTION_TO_POLARITY.get(emotion) if emotion is not None else None)

        if (i // batch_size) % 50 == 0:
            print(f"  Gemma ZS: {i + batch_size_cur}/{N}")

    del model, base_model
    if device == "cuda":
        import torch; torch.cuda.empty_cache()
    return preds


# ---------------------------------------------------------------------------
# Gemma inference — adapted (3-class output, no collapse needed)
# ---------------------------------------------------------------------------

def run_gemma_adapted(
    texts: List[str],
    base_probs_3: Dict[str, np.ndarray],
    weights: Dict[str, float],
    vsfc_lora_dir: Path,
    system_prompt: str,
    device: Optional[str] = None,
    batch_size: int = 4,
    max_new_tokens: int = 12,
) -> List[Optional[str]]:
    """Run VSFC-continued Gemma on texts; parse 3-class label directly.

    Returns list of predicted polarity strings, None for unparseable.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from tm_research.eval.vsfc_format_prompt import (
        format_prompt_3class,
        parse_label_3class,
        VSFC_LABEL_MAP,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(str(vsfc_lora_dir))
    base_model_name = _read_base_model_name(vsfc_lora_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map="auto", torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, str(vsfc_lora_dir))
    model.eval()

    usable = {k: v for k, v in weights.items() if k in base_probs_3}
    preds: List[Optional[str]] = []
    N = len(texts)

    for i in range(0, N, batch_size):
        batch_texts = texts[i: i + batch_size]
        batch_size_cur = len(batch_texts)
        prompts = []
        for j in range(batch_size_cur):
            row_probs = {m: base_probs_3[m][i + j] for m in usable}
            fp = format_prompt_3class(batch_texts[j], row_probs, usable, label=None)
            prompts.append(f"{system_prompt}\n\n{fp['prompt']}")

        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.eos_token_id, do_sample=False)
        inp_len = enc["input_ids"].shape[1]
        for seq in out:
            new_text = tok.decode(seq[inp_len:], skip_special_tokens=True)
            preds.append(parse_label_3class(new_text))

        if (i // batch_size) % 50 == 0:
            print(f"  Gemma Adapt: {i + batch_size_cur}/{N}")

    del model, base_model
    if device == "cuda":
        import torch; torch.cuda.empty_cache()
    return preds


# ---------------------------------------------------------------------------
# LoRA continuation on VSFC (Strategy 2 training)
# ---------------------------------------------------------------------------

def continue_lora_vsfc(
    train_jsonl: Path,
    val_jsonl: Path,
    output_dir: Path,
    base_model_name: str = _GEMMA_DEFAULT,
    num_epochs: int = 2,
    learning_rate: float = 5e-5,
    per_device_batch: int = 4,
    grad_accum: int = 4,
    max_seq_len: int = 512,
) -> None:
    """Continue LoRA fine-tuning from VSMEC adapter on VSFC JSONL.

    Saves the updated adapter to ``output_dir``. Requires TRL + PEFT + BitsAndBytes.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from tm_research.ensemble.utils_io import LORA_DIR as VSMEC_LORA_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(str(VSMEC_LORA_DIR))
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb_cfg, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(VSMEC_LORA_DIR), is_trainable=True)

    ds = load_dataset("json", data_files={
        "train": str(train_jsonl),
        "validation": str(val_jsonl),
    })

    def _fmt(example):
        return {"text": example["prompt"] + example["completion"]}

    ds = ds.map(_fmt)

    sft_cfg = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=max_seq_len,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    tok.save_pretrained(str(output_dir))
    print(f"  VSFC LoRA adapter saved → {output_dir}")

    # Save meta
    with open(output_dir / "vsfc_lora_train_meta.json", "w") as f:
        json.dump({
            "base_model": base_model_name,
            "continued_from": str(VSMEC_LORA_DIR),
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "system_prompt": VSFC_SYSTEM_PROMPT,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VSFC_SYSTEM_PROMPT = (
    "You are a sentiment classifier for Vietnamese student feedback text. "
    "Pick exactly one label from: negative, neutral, positive. "
    "You receive the input text and the per-class probability distributions of "
    "five base models (each with its weight). "
    "Output ONLY the final answer in the exact format: <label>LABEL</label>."
)

_GEMMA_DEFAULT = "google/gemma-2-9b-it"


def _read_base_model_name(lora_dir: Path, fallback: str = _GEMMA_DEFAULT) -> str:
    meta = lora_dir / "vsfc_lora_train_meta.json"
    if meta.exists():
        with open(meta) as f:
            return json.load(f).get("base_model", fallback)
    meta2 = lora_dir.parent / "lora_train_meta.json"
    if meta2.exists():
        with open(meta2) as f:
            return json.load(f).get("base_model", fallback)
    return fallback
