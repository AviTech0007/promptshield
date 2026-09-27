"""Signal 1: rules.

A library of regex patterns (patterns.yaml), each with a weight and a plain-English reason.
Runs in microseconds and every hit is explainable, which is why it exists.
It is NOT the main defence: rules are easy to reword around. That is what the
other signals and the action gate are for. Say exactly this if judges ask.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from ..types import Chunk, SignalScore

PATTERNS_FILE = Path(__file__).with_name("patterns.yaml")

# Being hidden from humans is suspicious on its own; hidden text that ALSO
# contains a command is very suspicious.
HIDDEN_BASE = 0.35
HIDDEN_WITH_HIT_FLOOR = 0.9


@lru_cache(maxsize=1)
def load_rules() -> list[dict]:
    data = yaml.safe_load(PATTERNS_FILE.read_text(encoding="utf-8"))
    rules = []
    for r in data["rules"]:
        rules.append({**r, "compiled": re.compile(r["regex"], re.IGNORECASE | re.DOTALL)})
    return rules


def score_chunk(chunk: Chunk) -> SignalScore:
    hits = [r for r in load_rules() if r["compiled"].search(chunk.text)]
    keep = 1.0
    for r in hits:
        w = float(r["weight"])
        if not r.get("standalone", True) and len(hits) == 1:
            w *= 0.5            # context rule on its own (e.g. "urgent", "don't tell anyone")
        keep *= (1.0 - w)
    score = 1.0 - keep
    reasons = [r["reason"] for r in hits]

    if chunk.hidden:
        reasons.insert(0, "Text is invisible to a human reader")
        score = max(score, HIDDEN_BASE)
        if hits:
            score = max(score, HIDDEN_WITH_HIT_FLOOR)

    return SignalScore(name="heuristics", score=round(score, 4), reasons=reasons)


def matched_rule_ids(text: str) -> list[str]:
    """Helper for tests and the dashboard."""
    return [r["id"] for r in load_rules() if r["compiled"].search(text)]
