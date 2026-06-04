"""UIT-VSFC 3-class sentiment label constants (UIT-VSFC HuggingFace card)."""

from __future__ import annotations

from typing import Dict, List

POLARITY_NEGATIVE = "negative"
POLARITY_NEUTRAL = "neutral"
POLARITY_POSITIVE = "positive"

POLARITY_CLASSES: List[str] = [POLARITY_NEGATIVE, POLARITY_NEUTRAL, POLARITY_POSITIVE]
POLARITY_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(POLARITY_CLASSES)}

VSFC_INT_TO_LABEL: Dict[int, str] = {
    0: POLARITY_NEGATIVE,
    1: POLARITY_NEUTRAL,
    2: POLARITY_POSITIVE,
}
