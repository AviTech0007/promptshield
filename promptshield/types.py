"""Data shapes shared by every module (pydantic models).

Keeping them in one file means the scanner, the gate, the API and the dashboard
all agree on exactly what a "scan result" or a "gate decision" looks like.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Level = Literal["allow", "warn", "quarantine"]
Decision = Literal["allow", "require_approval", "block"]


class Finding(BaseModel):
    """One thing the pre-processor noticed, e.g. hidden text or zero-width characters."""
    kind: str                       # e.g. "hidden_css", "zero_width", "base64", "exfil_image"
    detail: str                     # human-readable description
    text: str = ""                  # the extracted text, if any


class Chunk(BaseModel):
    """A sentence-sized piece of content that gets scored on its own."""
    index: int
    text: str
    hidden: bool = False            # True if this text was invisible to a human reader


class SignalScore(BaseModel):
    """Output of one detection signal for one chunk."""
    name: str                       # "heuristics" | "similarity" | "classifier" | "judge"
    score: float                    # 0.0 (clean) .. 1.0 (attack)
    available: bool = True          # False if the signal could not run (model missing etc.)
    reasons: list[str] = Field(default_factory=list)


class ChunkResult(BaseModel):
    chunk: Chunk
    signals: list[SignalScore]
    combined: float                 # 0..1 after fusion (and judge, if it ran)
    judged: bool = False


class ScanResult(BaseModel):
    """What shield.scan() returns."""
    taint_id: str
    source: str
    level: Level
    risk: int                       # 0..100
    clean_text: str                 # visible text, normalised, safe to show to the agent
    explanation: str
    findings: list[Finding]
    chunks: list[ChunkResult]
    top_chunk: Optional[ChunkResult] = None
    signal_summary: dict[str, Optional[float]]   # max score per signal (None = unavailable)
    latency_ms: float


class ActionRequest(BaseModel):
    """A tool call the agent wants to make."""
    tool: str
    args: dict[str, Any]
    session_id: str = "default"


class GateDecision(BaseModel):
    decision: Decision
    tool: str
    args: dict[str, Any]
    risk: str                       # "low" | "medium" | "high" | "critical"
    reasons: list[str]
    tainted_args: dict[str, str] = Field(default_factory=dict)   # arg -> taint_id it came from
    action_id: Optional[int] = None  # row id in the approval queue, if held
