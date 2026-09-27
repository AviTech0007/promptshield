"""The demo email agent, runnable with or without PromptShield.

Flow for each email (a common, realistic design for email agents):
    read email -> [PromptShield scan]  -> brain decides summary + actions
               -> for each action: [PromptShield action gate] -> run tool

Without the shield: the brain sees the raw email as a typical pipeline would (hidden text
included) and every action runs immediately. With the shield: content is scanned first
(quarantined emails are never shown to the brain, hidden text is stripped) and every tool
call is checked by the gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from promptshield import ActionHeld, Shield
from promptshield.config import POLICY_DIR
from promptshield.gate import Policy

DEFAULT_POLICY = POLICY_DIR / "email_agent.yaml"

from .brains import Brain
from .mailstore import MailStore, llm_view


@dataclass
class ActionRecord:
    tool: str
    args: dict
    outcome: str                      # "executed" | "held" | "blocked" | "error"
    detail: str = ""
    action_id: Optional[int] = None


@dataclass
class EmailStep:
    email_id: str
    sender: str
    subject: str
    kind: str
    scan: Any = None                  # ScanResult when shielded
    summary: str = ""
    actions: list[ActionRecord] = field(default_factory=list)


@dataclass
class AgentRun:
    shielded: bool
    brain: str
    task: str
    steps: list[EmailStep] = field(default_factory=list)
    leaked: list[dict] = field(default_factory=list)


class EmailAgent:
    def __init__(self, store: MailStore, brain: Brain, shield: Optional[Shield] = None,
                 session_id: str = "demo") -> None:
        self.store = store
        self.brain = brain
        self.shield = shield
        self.session_id = session_id
        self.tools = self._make_tools()

    # ------------------------------------------------------------------ tools
    def _make_tools(self) -> dict:
        store = self.store

        def forward_email(email_id: str, to: str, count: int = 1) -> str:
            return store.forward(email_id, to, int(count))

        def send_email(to: str, subject: str = "", body: str = "") -> str:
            return store.send(to, subject, body)

        def delete_email(email_id: str) -> str:
            return store.delete(email_id)

        tools = {"forward_email": forward_email, "send_email": send_email, "delete_email": delete_email}
        if self.shield is not None:
            tools = {name: self.shield.guard_tool(name=name, session_id=self.session_id)(fn)
                     for name, fn in tools.items()}
        return tools

    # ------------------------------------------------------------------ main loop
    def run(self, task: str) -> AgentRun:
        run = AgentRun(shielded=self.shield is not None, brain=self.brain.name, task=task)
        if self.shield is not None:
            self.shield.reset_session(self.session_id)
            self.shield.set_user_request(task, self.session_id)

        for email in self.store.inbox:
            step = EmailStep(email["id"], email["from"], email["subject"], email.get("kind", ""))
            content = self._content_for_brain(email, step)
            decision = self.brain.decide(task, self.store.user_email, email, content)
            step.summary = decision["summary"]
            for action in decision["actions"]:
                step.actions.append(self._run_action(email, action))
            run.steps.append(step)

        # Same trusted list in both modes, so "leaked" means the same thing before and after
        trusted = {self.store.user_email.lower()} | {v.lower() for v in Policy(DEFAULT_POLICY).trusted_values}
        run.leaked = self.store.leaked_to(trusted)
        return run

    def _content_for_brain(self, email: dict, step: EmailStep) -> str:
        if self.shield is None:
            return llm_view(email["body"])
        scan = self.shield.scan(email["body"], source=f"email {email['id']} from {email['from']}",
                                session_id=self.session_id, label=email["subject"])
        step.scan = scan
        if scan.level == "quarantine":
            return f"[Quarantined by PromptShield: {scan.explanation}]"
        return scan.clean_text

    def _run_action(self, email: dict, action: dict) -> ActionRecord:
        tool = action.get("tool", "")
        args = dict(action.get("args") or {})
        if tool not in self.tools:
            return ActionRecord(tool, args, "error", f"unknown tool {tool}")
        if tool in {"forward_email", "delete_email"}:
            args.setdefault("email_id", email["id"])
        try:
            out = self.tools[tool](**args, **({"_session_id": self.session_id} if self.shield else {}))
            return ActionRecord(tool, args, "executed", str(out))
        except ActionHeld as held:
            d = held.decision
            outcome = "blocked" if d.decision == "block" else "held"
            return ActionRecord(tool, args, outcome, "; ".join(d.reasons), d.action_id)
        except TypeError as exc:                    # model produced bad arguments
            return ActionRecord(tool, args, "error", str(exc))
