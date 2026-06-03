"""3-class prompt formatting for VSFC LoRA continuation and adapted Gemma inference.

Mirrors the schema from ``utils_stacking.format_prompt`` but uses a 3-class
label space (negative / neutral / positive) instead of the 7 VSMEC emotions.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 3-class label map (matches VSFC gold ordering)
# ---------------------------------------------------------------------------

POLARITY_CLASSES: List[str] = ["negative", "neutral", "positive"]
POLARITY_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(POLARITY_CLASSES)}


class _VSFCLabelMap:
    """Lightweight 3-class label map compatible with parse_label_from_completion."""

    label2id: Dict[str, int] = POLARITY_TO_IDX
    id2label: Dict[int, str] = dict(enumerate(POLARITY_CLASSES))
    num_classes: int = 3

    @property
    def class_names(self) -> List[str]:
        return POLARITY_CLASSES


VSFC_LABEL_MAP = _VSFCLabelMap()


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def _fmt_prob_dict(probs: np.ndarray, class_names: List[str], decimals: int = 3) -> str:
    parts = [f"{class_names[i]}={probs[i]:.{decimals}f}" for i in range(len(class_names))]
    return "{" + ", ".join(parts) + "}"


def format_prompt_3class(
    text: str,
    probs_per_model: Dict[str, np.ndarray],
    weights: Dict[str, float],
    label: Optional[str] = None,
    model_order: Iterable[str] = ("phobert", "cafebert", "vibert", "logreg", "svc"),
    class_names: List[str] = POLARITY_CLASSES,
) -> Dict[str, str]:
    """Render one VSFC stacking example as ``{prompt, completion}``.

    Prompt schema (3-class version):

        [TEXT] ... [/TEXT]
        [STACK]
        <phobert w=0.210>{negative=0.300, neutral=0.100, positive=0.600}
        ...
        [/STACK]
        [WEIGHTED_AVG]{negative=…, neutral=…, positive=…}[/WEIGHTED_AVG]

    Completion (only present when ``label`` is not None):
        ``<label>positive</label>``

    Parameters
    ----------
    text : input Vietnamese text.
    probs_per_model : ``{model_key: (3,) float array}`` — 3-class probs per model.
    weights : ``{model_key: float}`` — VSFC-tuned normalized weights.
    label : gold sentiment string; omit for inference rows.
    model_order : preferred order of models in the stack block.
    class_names : 3-class label names in prob-vector order.
    """
    used = [m for m in model_order if m in probs_per_model and m in weights]

    lines = [f"[TEXT] {text} [/TEXT]", "[STACK]"]
    for m in used:
        w = weights[m]
        lines.append(f"<{m} w={w:.3f}>{_fmt_prob_dict(probs_per_model[m], class_names)}")
    lines.append("[/STACK]")

    # Weighted average
    P = np.stack([probs_per_model[m] for m in used], axis=0)  # (M, 3)
    w_arr = np.array([weights[m] for m in used], dtype=np.float32)
    w_arr = w_arr / w_arr.sum()
    avg = P.T @ w_arr  # (3,)
    lines.append(f"[WEIGHTED_AVG]{_fmt_prob_dict(avg, class_names)}[/WEIGHTED_AVG]")

    prompt = "\n".join(lines)
    completion = f"<label>{label}</label>" if label is not None else ""
    return {"prompt": prompt, "completion": completion}


# ---------------------------------------------------------------------------
# Completion parser (3-class)
# ---------------------------------------------------------------------------

def parse_label_3class(text: str) -> Optional[str]:
    """Extract ``<label>...</label>`` from a Gemma completion (3-class version).

    Falls back to first polarity keyword found; returns None if not found.
    """
    m = re.search(r"<label>\s*([^<\n]+?)\s*</label>", text, flags=re.IGNORECASE)
    if m:
        cand = m.group(1).strip().lower()
        for lab in POLARITY_CLASSES:
            if lab == cand:
                return lab

    lower = text.lower()
    for lab in POLARITY_CLASSES:
        if lab in lower:
            return lab
    return None
