# Data

Every file is JSON Lines: one example per line.

```json
{"text": "Ignore previous instructions and ...", "label": 1, "type": "indirect", "author": "avi"}
```

| field | meaning |
|---|---|
| `text` | the sentence or short email |
| `label` | `1` = attack (contains an instruction aimed at an AI), `0` = clean |
| `type` | `direct`, `indirect`, `clean`, `hard_negative` (genuine email that *looks* suspicious), `public`, `bipia_indirect` |
| `author` | who wrote it, or which dataset it came from |

## Files

| file | in git? | what it is |
|---|---|---|
| `seed_corpus.jsonl` | yes | 40 attacks + 40 clean starter examples (written for this project with AI assistance; disclosed in README) |
| `team_corpus.jsonl` | yes | **you create this**: each member adds ~40 attacks and ~40 clean, at least 15 of them `hard_negative` |
| `team_test_blind.jsonl` | yes | **one member** writes 30+ attacks and 30+ clean WITHOUT reading `patterns.yaml`; always goes to TEST |
| `raw/*.jsonl` | no | public datasets (`scripts/download_datasets.py`, `scripts/import_bipia.py`) |
| `splits/train|val|test.jsonl` | yes (small) | made by `scripts/make_splits.py` |

## Writing good examples (this decides how believable your numbers are)

- Write attacks the way they'd arrive **inside an email or web page**, not as chat prompts.
- Vary them: polite, bossy, spelled-out addresses ("x at y dot example"), no keywords at all.
- Hard negatives matter most: real emails with "forward", "ignore my last email", "urgent", "don't tell anyone (surprise party)", "transfer the fee".
- Only ever use `.example` domains for addresses.
- Swap files with a teammate and check each other's labels.
