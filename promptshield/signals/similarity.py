"""Signal 2: similarity to known attacks.

Idea: an attack that has been reworded still MEANS the same thing, so it sits close to
known attacks in "meaning space". For each chunk we find its k nearest neighbours among
a library of labelled examples (known attacks + known clean text) and report:

    score = (similarity-weighted share of attacks among the neighbours)
            x (confidence: how close the nearest neighbour actually is)

so text unlike anything in the library scores low instead of a coin-flip 0.5.

Backends (picked automatically):
  * sentence-transformers all-MiniLM-L6-v2 embeddings  (best; needs requirements-ml.txt)
  * TF-IDF character n-grams from scikit-learn         (fallback; always available)

The library is the TRAIN split only (data/splits/train.jsonl), never val/test,
so evaluation numbers are not inflated by the answers being in the library.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import DATA_DIR, RUNTIME_DIR, SETTINGS
from ..types import Chunk, SignalScore

K = 5
CONFIDENT_SIM = {"embeddings": 0.55, "tfidf": 0.35}   # nearest-neighbour similarity that counts as "close"


def _library_path() -> Path:
    train = DATA_DIR / "splits" / "train.jsonl"
    return train if train.exists() else DATA_DIR / "seed_corpus.jsonl"


def _read_library(path: Path) -> tuple[list[str], np.ndarray]:
    texts, labels = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, np.array(labels)


class _Index:
    def __init__(self) -> None:
        path = _library_path()
        self.texts, self.labels = _read_library(path)
        self.backend = "tfidf"
        self._model = None
        if not SETTINGS.disable_embeddings:
            try:
                from sentence_transformers import SentenceTransformer  # heavy import
                self._model = SentenceTransformer(SETTINGS.embed_model)
                self.backend = "embeddings"
            except Exception:
                self._model = None
        if self.backend == "embeddings":
            self.matrix = self._cached_embeddings(path)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True,
                                        sublinear_tf=True, min_df=1)
            self.matrix = self._vec.fit_transform(self.texts)

    def _cached_embeddings(self, path: Path) -> np.ndarray:
        key = hashlib.sha1((path.read_bytes() + SETTINGS.embed_model.encode())).hexdigest()[:16]
        cache = RUNTIME_DIR / f"sim_cache_{key}.npy"
        if cache.exists():
            return np.load(cache)
        emb = self._model.encode(self.texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        np.save(cache, emb)
        return emb

    def similarities(self, texts: list[str]) -> np.ndarray:
        if self.backend == "embeddings":
            q = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return q @ self.matrix.T
        q = self._vec.transform(texts)
        return (q @ self.matrix.T).toarray()


@lru_cache(maxsize=1)
def get_index() -> _Index:
    return _Index()


def reset_index() -> None:
    """Call after changing the library (e.g. after make_splits.py) in a long-running process."""
    get_index.cache_clear()


def score_chunks(chunks: list[Chunk]) -> list[SignalScore]:
    if not chunks:
        return []
    idx = get_index()
    sims = idx.similarities([c.text for c in chunks])
    out = []
    for i, _ in enumerate(chunks):
        row = sims[i]
        top = np.argsort(-row)[:K]
        top_sims = np.clip(row[top], 0, None)
        weights = top_sims + 1e-6
        attack_share = float((weights * idx.labels[top]).sum() / weights.sum())
        confidence = float(min(1.0, top_sims[0] / CONFIDENT_SIM[idx.backend]))
        score = attack_share * confidence
        n_attacks = int(idx.labels[top].sum())
        reasons = []
        if score >= 0.5:
            nearest_attack = next((idx.texts[j] for j in top if idx.labels[j] == 1), "")
            reasons.append(f"{n_attacks} of {K} most similar known examples are attacks"
                           + (f' (closest: "{nearest_attack[:80]}")' if nearest_attack else ""))
        out.append(SignalScore(name="similarity", score=round(score, 4), reasons=reasons))
    return out


def backend_name() -> str:
    return get_index().backend
