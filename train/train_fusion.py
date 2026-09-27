"""Learn how much to trust each fast signal (logistic regression on the VAL split).

    python -m train.train_fusion

Why: the hand-set weights in config.py are guesses. Here we run the fast signals on
labelled validation examples and fit a small logistic regression:

    P(attack) = sigmoid(b + w1*rules + w2*similarity + w3*classifier)

The learned numbers are saved to runtime/fusion_weights.json; fuse.py uses them
automatically whenever the same signals are available. Re-run after you add data or
change a signal. We fit on VAL (not TRAIN) because the similarity signal already uses
TRAIN as its library, so its scores on TRAIN would be unrealistically perfect.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from promptshield import fuse                              # noqa: E402
from promptshield.config import DATA_DIR                   # noqa: E402
from promptshield.preprocess import preprocess             # noqa: E402
from promptshield.signals import classifier, heuristics, similarity   # noqa: E402


def features(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    names = ["heuristics", "similarity"] + (["classifier"] if classifier.is_available() else [])
    rows = []
    for t in texts:
        chunks = preprocess(t).chunks
        if not chunks:
            rows.append([0.0] * len(names))
            continue
        feats = {"heuristics": max(heuristics.score_chunk(c).score for c in chunks),
                 "similarity": max(s.score for s in similarity.score_chunks(chunks))}
        if "classifier" in names:
            feats["classifier"] = max(s.score for s in classifier.score_chunks(chunks))
        rows.append([feats[n] for n in names])
    return np.array(rows), names


def main() -> None:
    val_path = DATA_DIR / "splits" / "val.jsonl"
    if not val_path.exists():
        raise SystemExit("Run `python scripts/make_splits.py` first.")
    rows = [json.loads(l) for l in val_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts, y = [r["text"] for r in rows], np.array([int(r["label"]) for r in rows])
    if len(set(y)) < 2:
        raise SystemExit("VAL split needs both attacks and clean examples.")

    X, names = features(texts)
    print(f"Fitting on {len(y)} validation examples with signals: {names}")
    model = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000).fit(X, y)

    before = [1 if fuse.fast_score_from_values(dict(zip(names, x))) >= 0.4 else 0 for x in X]
    after = (model.predict_proba(X)[:, 1] >= 0.4).astype(int)
    print(f"val F1 with hand weights:    {f1_score(y, before):.3f}")
    print(f"val F1 with learned weights: {f1_score(y, after):.3f}  (optimistic: same data)")

    out = {"signals": names, "coef": dict(zip(names, map(float, model.coef_[0]))),
           "intercept": float(model.intercept_[0]), "n_train": int(len(y))}
    fuse.WEIGHTS_FILE.write_text(json.dumps(out, indent=2))
    fuse.reload_learned_model()
    print(f"saved -> {fuse.WEIGHTS_FILE.relative_to(ROOT)}")
    print(json.dumps(out["coef"], indent=2))
    if len(y) < fuse.MIN_TRAIN_FOR_LEARNED:
        print(f"\nNOTE: only {len(y)} validation examples. fuse.py ignores learned weights below "
              f"{fuse.MIN_TRAIN_FOR_LEARNED} and keeps the hand-set weights from config.py.\n"
              "Add public datasets + team examples, re-run make_splits.py, then retrain.")


if __name__ == "__main__":
    main()
