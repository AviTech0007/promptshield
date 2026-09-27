"""Download public prompt-injection datasets into data/raw/ as JSONL (text, label, type, author).

    python scripts/download_datasets.py

Needs `pip install datasets` (included in requirements-ml.txt) and internet.
Each dataset is written to its own file so you can drop any that cause trouble.
CHECK EACH LICENSE on its Hugging Face page and record it in data/LICENSES.md.

BIPIA (Microsoft's indirect prompt injection benchmark) lives on GitHub, not the HF hub:
    git clone https://github.com/microsoft/BIPIA data/raw/BIPIA
then run:  python scripts/import_bipia.py
"""
from __future__ import annotations

import json
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# (hf name, split, text column, label column, how to read the label)
SOURCES = [
    ("deepset/prompt-injections", "train", "text", "label", lambda v: int(v)),
    ("deepset/prompt-injections", "test", "text", "label", lambda v: int(v)),
    ("protectai/prompt-injection-validation", None, "text", "label", lambda v: int(v)),
]


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets   (or: pip install -r requirements-ml.txt)")

    for name, split, text_col, label_col, to_label in SOURCES:
        out = RAW / (name.replace("/", "__") + (f"__{split}" if split else "") + ".jsonl")
        try:
            ds = load_dataset(name, split=split) if split else load_dataset(name)
        except Exception as exc:
            print(f"SKIP {name}: {exc.__class__.__name__}: {str(exc)[:150]}")
            continue
        splits = [ds] if split else [ds[k] for k in ds.keys()]
        n = 0
        with out.open("w", encoding="utf-8") as f:
            for part in splits:
                cols = part.column_names
                tc = text_col if text_col in cols else next((c for c in cols if "text" in c or "prompt" in c), None)
                lc = label_col if label_col in cols else next((c for c in cols if "label" in c), None)
                if tc is None or lc is None:
                    print(f"SKIP {name}: can't find text/label columns in {cols}")
                    break
                for row in part:
                    text = str(row[tc]).strip()
                    if not text:
                        continue
                    f.write(json.dumps({"text": text, "label": to_label(row[lc]),
                                        "type": "public", "author": name}) + "\n")
                    n += 1
        print(f"wrote {n:5d} rows -> {out.relative_to(RAW.parent.parent)}")


if __name__ == "__main__":
    main()
