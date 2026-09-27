"""Build train / val / test splits from everything in data/.

    python scripts/make_splits.py

Inputs:
  data/seed_corpus.jsonl              hand-written starter set (in the repo)
  data/team_corpus.jsonl              YOUR examples (each member adds ~40 attacks + ~40 clean)
  data/team_test_blind.jsonl          attacks written by ONE teammate WITHOUT looking at the rules;
                                      these always go to TEST only (keeps the numbers honest)
  data/raw/*.jsonl                    public datasets (scripts/download_datasets.py)

Outputs: data/splits/train.jsonl (60%), val.jsonl (20%), test.jsonl (20%), stratified by label+type.

Rules that keep evaluation honest (say these to judges):
  * the similarity library and the learned weights only ever see TRAIN / VAL;
  * TEST is used once, by eval/run_eval.py;
  * duplicates (after lower-casing and squashing spaces) are removed across all splits,
    so the same sentence can't be in train and test.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "splits"
MAX_PER_PUBLIC_FILE = 1500          # keep public data from drowning out the email-style examples


def _read(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            r["label"] = int(r["label"])
            rows.append(r)
    return rows


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def main(seed: int = 7) -> None:
    rng = random.Random(seed)
    pool: list[dict] = []
    for name in ("seed_corpus.jsonl", "team_corpus.jsonl"):
        if (DATA / name).exists():
            pool += _read(DATA / name)
    for path in sorted((DATA / "raw").glob("*.jsonl")):
        rows = _read(path)
        rng.shuffle(rows)
        pool += rows[:MAX_PER_PUBLIC_FILE]

    blind = _read(DATA / "team_test_blind.jsonl") if (DATA / "team_test_blind.jsonl").exists() else []
    seen = {_key(r["text"]) for r in blind}
    unique = []
    for r in pool:
        k = _key(r["text"])
        if k not in seen:
            seen.add(k)
            unique.append(r)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in unique:
        groups[(r["label"], r.get("type", "?"))].append(r)
    train, val, test = [], [], list(blind)
    for rows in groups.values():
        rng.shuffle(rows)
        n = len(rows)
        if n < 3:                   # too few to split; keep for training
            train += rows
            continue
        n_test = max(1, round(n * 0.2))
        n_val = max(1, round(n * 0.2))
        test += rows[:n_test]
        val += rows[n_test:n_test + n_val]
        train += rows[n_test + n_val:]

    OUT.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        rng.shuffle(rows)
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_att = sum(r["label"] for r in rows)
        print(f"{name:5s}: {len(rows):5d} rows  ({n_att} attacks, {len(rows) - n_att} clean)")


if __name__ == "__main__":
    main()
