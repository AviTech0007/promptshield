"""Signal 3: an open-source prompt-injection classifier.

Default: protectai/deberta-v3-base-prompt-injection-v2 (downloads without approval).
Alternative: meta-llama/Llama-Prompt-Guard-2-22M or -86M (accept Meta's license on
Hugging Face first, then `huggingface-cli login`).

We use it as ONE input and then show (eval/run_eval.py ablation) that the full system
beats it on its own. That comparison is the answer to "isn't this just a wrapper?".

These models read at most 512 tokens. preprocess.py already cuts content into ~100-token
chunks, so nothing in a long email is silently skipped.

If transformers/torch are not installed or the model can't load, the signal reports
available=False and fuse.py re-balances the other weights. The app keeps working.
"""
from __future__ import annotations

from functools import lru_cache

from ..config import SETTINGS
from ..types import Chunk, SignalScore

ATTACK_LABELS = {"INJECTION", "MALICIOUS", "JAILBREAK", "LABEL_1", "UNSAFE"}


@lru_cache(maxsize=1)
def _pipeline():
    if SETTINGS.disable_classifier:
        return None
    try:
        from transformers import pipeline
        return pipeline("text-classification", model=SETTINGS.classifier_model,
                        truncation=True, max_length=512)
    except Exception as exc:          # not installed, no internet, gated model, ...
        print(f"[promptshield] classifier unavailable: {exc.__class__.__name__}: {str(exc)[:120]}")
        return None


def is_available() -> bool:
    return _pipeline() is not None


def score_chunks(chunks: list[Chunk]) -> list[SignalScore]:
    pipe = _pipeline()
    if pipe is None:
        return [SignalScore(name="classifier", score=0.0, available=False,
                            reasons=["classifier not loaded"]) for _ in chunks]
    preds = pipe([c.text for c in chunks], batch_size=16)
    out = []
    for p in preds:
        label = str(p["label"]).upper()
        prob = float(p["score"])
        score = prob if label in ATTACK_LABELS else 1.0 - prob
        reasons = [f"Injection classifier: {score:.0%} likely an injection"] if score >= 0.5 else []
        out.append(SignalScore(name="classifier", score=round(score, 4), reasons=reasons))
    return out
