"""Checkpoint 2: the action gate.

Every tool call the agent wants to make passes through `ActionGate.decide()` BEFORE it runs.
The decision depends on three questions:

  1. How risky is this tool?               (from the policy YAML)
  2. Did any destination argument come from untrusted content the user never mentioned?
                                           (taint check, see taint.py)
  3. Is flagged (warn/quarantine) content already in this conversation?

Result: allow / require_approval / block, with human-readable reasons.
This is what stops an attack even when every detector misses it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .taint import SessionTaint
from .types import GateDecision


class Policy:
    def __init__(self, path: Path | str) -> None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.default_risk: str = data.get("default_risk", "high")
        self.tools: dict[str, dict] = data.get("tools", {})
        self.trusted_values: list[str] = data.get("trusted_values", [])
        self.bulk_threshold: int = int(data.get("bulk_threshold", 5))

    def tool(self, name: str) -> dict:
        return self.tools.get(name, {"risk": self.default_risk})


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


class ActionGate:
    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def decide(self, tool: str, args: dict[str, Any], taint: SessionTaint) -> GateDecision:
        spec = self.policy.tool(tool)
        risk = spec.get("risk", self.policy.default_risk)
        reasons: list[str] = []
        tainted: dict[str, str] = {}
        from_quarantined = False

        # ---- (2) taint check on destination arguments -------------------------
        watch = spec.get("watch_args", [])
        for arg in watch:
            for value in _as_strings(args.get(arg)):
                rec = taint.origin_of(value)
                if rec is not None:
                    tainted[arg] = rec.taint_id
                    from_quarantined |= rec.level == "quarantine"
                    reasons.append(f'"{arg}={value}" appears only in untrusted {rec.source} '
                                   f"({rec.taint_id}, risk {rec.risk}), not in your request or contacts")

        # ---- bulk actions ---------------------------------------------------
        bulk = False
        for arg in spec.get("bulk_args", []):
            v = args.get(arg)
            if v is True or (isinstance(v, (int, float)) and v >= self.policy.bulk_threshold) \
                    or (isinstance(v, str) and v.lower() in {"all", "*", "every"}):
                bulk = True
                reasons.append(f"Bulk action ({arg}={v})")

        # ---- (3) flagged content in the conversation -------------------------
        flagged = taint.flagged_records()
        if flagged and risk in {"high", "critical"}:
            worst = max(flagged, key=lambda r: r.risk)
            reasons.append(f"Suspicious content reached the agent earlier ({worst.source}, risk {worst.risk})")

        # ---- (1) decide by risk level -----------------------------------------
        if risk == "low":
            decision = "allow"
            reasons = reasons or ["Read-only tool"]
        elif risk == "critical":
            decision = "block" if from_quarantined else "require_approval"
            reasons.insert(0, "Critical tool: always needs a human")
        elif risk == "medium":
            decision = "require_approval" if tainted else "allow"
        else:  # high
            if tainted and from_quarantined:
                decision = "block"
            elif tainted or flagged or (bulk and taint.records):
                decision = "require_approval"
            else:
                decision = "allow"

        if decision == "allow" and not reasons:
            reasons = ["No untrusted arguments and no flagged content"]
        return GateDecision(decision=decision, tool=tool, args=args, risk=risk,
                            reasons=reasons, tainted_args=tainted)
