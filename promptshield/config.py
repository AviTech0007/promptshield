"""Central settings for PromptShield.

Every tunable number lives here so the team changes it in ONE place.
Values can be overridden with environment variables (or a .env file in the repo root).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # repo root
RUNTIME_DIR = ROOT / "runtime"                          # SQLite files, caches (git-ignored)
DATA_DIR = ROOT / "data"
POLICY_DIR = ROOT / "policies"


def _load_dotenv(path: Path) -> None:
    """Tiny .env reader so we don't need an extra dependency. Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip() in {"1", "true", "yes"}


@dataclass
class Settings:
    # ---- models -------------------------------------------------------------
    ollama_url: str = field(default_factory=lambda: _env("PS_OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("PS_OLLAMA_MODEL", "qwen2.5:3b"))
    groq_api_key: str = field(default_factory=lambda: _env("PS_GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: _env("PS_GROQ_MODEL", "llama-3.1-8b-instant"))
    classifier_model: str = field(
        default_factory=lambda: _env("PS_CLASSIFIER_MODEL", "protectai/deberta-v3-base-prompt-injection-v2"))
    embed_model: str = field(
        default_factory=lambda: _env("PS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    disable_classifier: bool = field(default_factory=lambda: _flag("PS_DISABLE_CLASSIFIER"))
    disable_embeddings: bool = field(default_factory=lambda: _flag("PS_DISABLE_EMBEDDINGS"))
    disable_judge: bool = field(default_factory=lambda: _flag("PS_DISABLE_JUDGE"))

    # ---- chunking -----------------------------------------------------------
    max_chunk_chars: int = 400          # ~100 tokens; well under the classifier's 512-token limit

    # ---- score combining ----------------------------------------------------
    # Hand-set starting weights. train/train_fusion.py learns better ones and writes
    # runtime/fusion_weights.json, which fuse.py loads automatically if present.
    weights: dict = field(default_factory=lambda: {
        "heuristics": 0.35,
        "similarity": 0.25,
        "classifier": 0.40,
    })
    # A single very strong signal should not be averaged away.
    strong_signal_floor: float = 0.90   # if any signal >= this, score is at least this * 100 * 0.8

    # Unclear band: only chunks whose combined score lands here are sent to the LLM judge.
    judge_band: tuple = (0.30, 0.70)
    judge_weight: float = 0.5           # how much the judge can move the final score

    # Levels on the 0-100 scale
    warn_at: int = 40
    quarantine_at: int = 70

    # ---- timeouts -----------------------------------------------------------
    llm_timeout_s: float = 30.0


SETTINGS = Settings()
RUNTIME_DIR.mkdir(exist_ok=True)
