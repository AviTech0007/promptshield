"""HTTP API so agents written in ANY language (JS, Go, ...) can use PromptShield.

Run:   uvicorn api.main:app --reload --port 8000
Docs:  http://localhost:8000/docs   (interactive; great to show judges)

Flow for a client agent:
  1. POST /session/user_request   - tell the gate what the user actually asked
  2. POST /scan                   - before putting any untrusted content in the prompt
  3. POST /check_action           - before running any tool; obey the decision
  4. GET  /actions/pending, POST /actions/{id}/approve|deny   - human approval
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from promptshield import Shield
from promptshield.signals import classifier, similarity
from promptshield.types import GateDecision, ScanResult

app = FastAPI(title="PromptShield API", version="0.9.0",
              description="Two-checkpoint firewall for AI agents: scan content in, gate actions out.")
shield = Shield()


class ScanIn(BaseModel):
    content: str
    source: str = "content"
    session_id: str = "default"


class ActionIn(BaseModel):
    tool: str
    args: dict[str, Any]
    session_id: str = "default"


class UserRequestIn(BaseModel):
    text: str
    session_id: str = "default"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "similarity_backend": similarity.backend_name(),
            "classifier_loaded": classifier.is_available(), "judge_enabled": shield.use_judge}


@app.post("/session/user_request")
def user_request(body: UserRequestIn) -> dict:
    shield.set_user_request(body.text, body.session_id)
    return {"ok": True}


@app.post("/scan", response_model=ScanResult)
def scan(body: ScanIn) -> ScanResult:
    return shield.scan(body.content, source=body.source, session_id=body.session_id)


@app.post("/check_action", response_model=GateDecision)
def check_action(body: ActionIn) -> GateDecision:
    return shield.check_action(body.tool, body.args, body.session_id)


@app.get("/actions/pending")
def pending() -> list[dict]:
    return shield.audit.pending_actions()


@app.post("/actions/{action_id}/approve")
def approve(action_id: int, by: str = "api-user") -> dict:
    try:
        # execute=False: the tool lives in the client's process; the client runs it after approval
        shield.approve(action_id, by=by, execute=False)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"id": action_id, "status": "approved"}


@app.post("/actions/{action_id}/deny")
def deny(action_id: int, by: str = "api-user") -> dict:
    shield.deny(action_id, by=by)
    return {"id": action_id, "status": "denied"}


@app.get("/audit/scans")
def audit_scans(limit: int = 50) -> list[dict]:
    rows = shield.audit.recent_scans(limit)
    for r in rows:
        r.pop("result_json", None)
    return rows


@app.get("/audit/actions")
def audit_actions(limit: int = 50) -> list[dict]:
    return shield.audit.recent_actions(limit)
