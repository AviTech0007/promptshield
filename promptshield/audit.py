"""Audit log + approval queue, stored in SQLite (one file, no server).

Two tables:
  scans    - every piece of content checked, with its risk score and explanation
  actions  - every tool call the agent tried, the gate's decision, and what happened next
             (status: allowed -> executed, pending -> approved/denied -> executed, blocked)

The dashboard reads these tables; judges can open runtime/promptshield.db to verify.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .config import RUNTIME_DIR

DEFAULT_DB = RUNTIME_DIR / "promptshield.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL, session TEXT, taint_id TEXT, source TEXT, label TEXT,
  level TEXT, risk INTEGER, explanation TEXT, result_json TEXT
);
CREATE TABLE IF NOT EXISTS actions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL, session TEXT, tool TEXT, args_json TEXT, risk TEXT,
  decision TEXT, reasons_json TEXT, tainted_json TEXT,
  status TEXT, decided_by TEXT, decided_at REAL, result TEXT
);
"""


class AuditLog:
    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.path = Path(db_path) if db_path else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ scans
    def log_scan(self, session: str, result, label: str = "") -> int:
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO scans (ts, session, taint_id, source, label, level, risk, explanation, result_json)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), session, result.taint_id, result.source, label, result.level, result.risk,
                 result.explanation, result.model_dump_json()))
            return cur.lastrowid

    def recent_scans(self, limit: int = 100) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ actions
    def log_action(self, session: str, decision, status: str) -> int:
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO actions (ts, session, tool, args_json, risk, decision, reasons_json, tainted_json, status)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), session, decision.tool, json.dumps(decision.args, default=str), decision.risk,
                 decision.decision, json.dumps(decision.reasons), json.dumps(decision.tainted_args), status))
            return cur.lastrowid

    def set_status(self, action_id: int, status: str, by: str = "", result: Any = None) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE actions SET status=?, decided_by=COALESCE(NULLIF(?, ''), decided_by),"
                      " decided_at=?, result=COALESCE(?, result) WHERE id=?",
                      (status, by, time.time(), None if result is None else str(result)[:2000], action_id))

    def get_action(self, action_id: int) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM actions WHERE id=?", (action_id,)).fetchone()
        return dict(row) if row else None

    def pending_actions(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM actions WHERE status='pending' ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def recent_actions(self, limit: int = 100) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM actions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def clear(self) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM scans")
            c.execute("DELETE FROM actions")
