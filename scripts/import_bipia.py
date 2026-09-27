"""Turn Microsoft BIPIA's attack instructions into labelled indirect-injection examples.

    git clone https://github.com/microsoft/BIPIA data/raw/BIPIA
    python scripts/import_bipia.py

BIPIA ships lists of attack instructions (text and code attacks). We embed each one inside
a harmless email-style context, which is how indirect injection actually arrives, and write
them to data/raw/bipia_email_attacks.jsonl with label 1.

The folder layout of the BIPIA repo can change; this script searches for its attack JSON
files instead of assuming exact paths. If it finds nothing, open the repo and point
ATTACK_GLOBS at the right files.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIPIA = ROOT / "data" / "raw" / "BIPIA"
OUT = ROOT / "data" / "raw" / "bipia_email_attacks.jsonl"
ATTACK_GLOBS = ["**/*attack*.json", "**/*attack*.jsonl"]

CONTEXTS = [
    "Hi, following up on the order from last week. {attack} Thanks, and let me know about delivery dates.",
    "Please find the meeting notes below. {attack} Next meeting is on Tuesday.",
    "Your subscription renews next month. {attack} You can manage billing in your account settings.",
    "Hello team, the report is attached for review. {attack} Regards, Operations",
]


def _collect(obj, out: list[str]) -> None:
    if isinstance(obj, str) and 4 <= len(obj.split()) <= 80:
        out.append(obj.strip())
    elif isinstance(obj, list):
        for x in obj:
            _collect(x, out)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, out)


def main() -> None:
    if not BIPIA.exists():
        raise SystemExit("First: git clone https://github.com/microsoft/BIPIA data/raw/BIPIA")
    attacks: list[str] = []
    for pattern in ATTACK_GLOBS:
        for path in BIPIA.glob(pattern):
            try:
                text = path.read_text(encoding="utf-8")
                data = [json.loads(l) for l in text.splitlines() if l.strip()] if path.suffix == ".jsonl" \
                    else json.loads(text)
            except Exception:
                continue
            _collect(data, attacks)
    attacks = sorted(set(attacks))
    rng = random.Random(7)
    with OUT.open("w", encoding="utf-8") as f:
        for a in attacks:
            f.write(json.dumps({"text": rng.choice(CONTEXTS).format(attack=a), "label": 1,
                                "type": "bipia_indirect", "author": "microsoft/BIPIA"}) + "\n")
    print(f"wrote {len(attacks)} BIPIA-based examples -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
