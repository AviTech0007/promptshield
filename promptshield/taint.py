"""Taint tracking: remember which values came from untrusted content.

When shield.scan() lets content through to the agent, it is "tainted" (untrusted).
We pull out the values an attacker would want the agent to use: email addresses,
URLs/domains, phone and account numbers. Later, when the agent calls a tool, the
action gate checks each argument:

    "Does this value appear in untrusted content, but NOT in anything the user
     themselves said or in the trusted contacts list?"

If yes, the instruction to use it almost certainly came from the content, not the user.
That check doesn't care how the attack was worded, which is why it still works
when every detector misses a rephrased attack.

We also undo simple obfuscation ("billing dash verify at mailbox dot example") before
comparing, so spelling an address out in words doesn't slip past.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

_EMAIL = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", re.I)
_URL = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
_DOMAIN = re.compile(r"\b(?:[a-z0-9\-]+\.)+(?:com|net|org|io|in|co|example|xyz|ru|info|biz|dev|app|me)\b", re.I)
_NUMBER = re.compile(r"\+?\d[\d\s\-]{6,}\d")

_OBFUSCATIONS = [
    (re.compile(r"\s*[\[\(\{<]\s*(at|@)\s*[\]\)\}>]\s*", re.I), "@"),
    (re.compile(r"\s*[\[\(\{<]\s*(dot|\.)\s*[\]\)\}>]\s*", re.I), "."),
    (re.compile(r"\s+at\s+(?=[a-z0-9\-]+\s+dot\s+)", re.I), "@"),
    (re.compile(r"\s+dot\s+(?=[a-z]{2,}\b)", re.I), "."),
    (re.compile(r"\s+(dash|hyphen)\s+", re.I), "-"),
    (re.compile(r"\s+underscore\s+", re.I), "_"),
]


def deobfuscate(text: str) -> str:
    out = text
    for pattern, repl in _OBFUSCATIONS:
        out = pattern.sub(repl, out)
    return out


def _norm_url(url: str) -> str:
    return url.lower().rstrip("/.,;")


def extract_values(text: str) -> set[str]:
    """Pull out emails, URLs, domains and long numbers, normalised for comparison."""
    text = deobfuscate(text)
    values: set[str] = set()
    for m in _EMAIL.findall(text):
        values.add(m.lower())
        values.add(m.lower().split("@", 1)[1])            # the domain too
    for m in _URL.findall(text):
        values.add(_norm_url(m))
        host = re.sub(r"^https?://", "", m.lower()).split("/")[0]
        values.add(host)
    for m in _DOMAIN.findall(text):
        values.add(m.lower())
    for m in _NUMBER.findall(text):
        digits = re.sub(r"\D", "", m)
        if len(digits) >= 7:
            values.add(digits)
    return values


@dataclass
class TaintRecord:
    taint_id: str
    source: str
    level: str
    risk: int
    values: set[str]


@dataclass
class SessionTaint:
    """Everything the gate needs to know about one agent session."""
    records: dict[str, TaintRecord] = field(default_factory=dict)
    trusted: set[str] = field(default_factory=set)       # values from the user's own words / contacts

    def add_untrusted(self, text: str, source: str, level: str, risk: int) -> str:
        taint_id = "t_" + uuid.uuid4().hex[:8]
        self.records[taint_id] = TaintRecord(taint_id, source, level, risk, extract_values(text))
        return taint_id

    def add_trusted(self, text_or_values) -> None:
        if isinstance(text_or_values, str):
            self.trusted |= extract_values(text_or_values)
            self.trusted.add(text_or_values.strip().lower())
        else:
            for v in text_or_values:
                self.trusted |= extract_values(v) | {v.strip().lower()}

    def origin_of(self, value: str) -> TaintRecord | None:
        """Which untrusted record introduced this value (and the user never mentioned it)?"""
        candidates = extract_values(value) or {value.strip().lower()}
        for v in candidates:
            if v in self.trusted:
                continue
            for rec in self.records.values():
                if v in rec.values:
                    return rec
        return None

    def flagged_records(self) -> list[TaintRecord]:
        """Suspicious content that still REACHED the agent (level 'warn').
        Quarantined content never reaches the agent, so it can't steer later actions;
        its values are still tainted, which is what the argument check uses."""
        return [r for r in self.records.values() if r.level == "warn"]
