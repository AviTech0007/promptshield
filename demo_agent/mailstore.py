"""Mock mailbox for the demo: an inbox loaded from inbox.json and an outbox in memory.

NOTHING here sends real email. `forward_email` / `send_email` only append to the outbox
list, and every address uses the reserved `.example` domain. Say this in the README:
we only ever attack our own mock agent.

Also contains `llm_view()`: how a typical (unprotected) agent pipeline turns an email into
text for the LLM. It strips HTML tags but KEEPS hidden text and comments, and invisible
Unicode tag characters survive too. That is exactly why hidden instructions work.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Comment

INBOX_FILE = Path(__file__).with_name("inbox.json")
_TAGS = re.compile(r"\{\{TAGS:(.*?)\}\}", re.S)


def encode_tags(text: str) -> str:
    """Replace {{TAGS:msg}} with invisible Unicode tag characters (U+E0000 block)."""
    return _TAGS.sub(lambda m: "".join(chr(0xE0000 + ord(ch)) for ch in m.group(1)), text)


def llm_view(raw: str) -> str:
    """What a naive agent pipeline hands to its LLM: tags removed, hidden text kept."""
    if "<" in raw and ">" in raw:
        soup = BeautifulSoup(raw, "html.parser")
        for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
            c.replace_with(" " + str(c) + " ")        # comments end up in the text
        text = soup.get_text(" ")
    else:
        text = raw
    # LLM tokenizers can read Unicode tag characters; we decode them to show what the model "sees"
    text = "".join(chr(ord(ch) - 0xE0000) if 0xE0020 <= ord(ch) < 0xE007F else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class MailStore:
    inbox: list[dict] = field(default_factory=list)
    outbox: list[dict] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    user_email: str = "avi@acme.example"

    @classmethod
    def load(cls, path: Path = INBOX_FILE, only_ids: list[str] | None = None) -> "MailStore":
        data = json.loads(path.read_text(encoding="utf-8"))
        emails = []
        for e in data["emails"]:
            if only_ids and e["id"] not in only_ids:
                continue
            e = copy.deepcopy(e)
            e["body"] = encode_tags(e["body"])
            emails.append(e)
        return cls(inbox=emails, user_email=data.get("user_email", "avi@acme.example"))

    def get(self, email_id: str) -> dict | None:
        return next((e for e in self.inbox if e["id"] == email_id), None)

    # ---- the "tools" (plain functions; the agent wraps them) -----------------
    def list_emails(self) -> list[dict]:
        return [{"id": e["id"], "from": e["from"], "subject": e["subject"]} for e in self.inbox]

    def forward(self, email_id: str, to: str, count: int = 1) -> str:
        """Forward `email_id`, or the most recent `count` emails if count > 1."""
        ids = [e["id"] for e in self.inbox][-count:] if count > 1 else [email_id]
        for i in ids:
            e = self.get(i)
            if e:
                self.outbox.append({"to": to, "subject": "Fwd: " + e["subject"], "email_id": i})
        return f"Forwarded {len(ids)} email(s) to {to}"

    def send(self, to: str, subject: str, body: str) -> str:
        self.outbox.append({"to": to, "subject": subject, "body": body[:500]})
        return f"Sent '{subject}' to {to}"

    def delete(self, email_id: str) -> str:
        self.deleted.append(email_id)
        return f"Deleted {email_id}"

    def leaked_to(self, trusted: set[str]) -> list[dict]:
        """Outbox entries whose recipient is not trusted = data that left to an attacker."""
        return [m for m in self.outbox if m["to"].lower() not in trusted]
