# Data, model and library disclosure

The ASYNC'26 rules require disclosing every dataset, model, API and library, with its licence.
**Check each licence yourself on its official page before the final submission** and fix this table.

## Datasets

| name | source | licence (verify!) | how we use it |
|---|---|---|---|
| Seed corpus | this repo, `data/seed_corpus.jsonl` | MIT (ours) | starter train/val/test examples; written with AI assistance and reviewed by the team |
| Team corpus | this repo | MIT (ours) | examples written by team members |
| deepset/prompt-injections | huggingface.co/datasets/deepset/prompt-injections | check page | training + test |
| protectai/prompt-injection-validation | huggingface.co/datasets/protectai/prompt-injection-validation | check page | test |
| Microsoft BIPIA | github.com/microsoft/BIPIA | check repo LICENSE | indirect-injection attacks wrapped in email text |

## Models

| model | licence (verify!) | used for |
|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 | similarity signal |
| protectai/deberta-v3-base-prompt-injection-v2 | check model card | classifier signal |
| meta-llama/Llama-Prompt-Guard-2-22M / 86M (optional) | Llama 4 Community License (gated) | classifier signal |
| Qwen2.5 3B via Ollama (or whichever you pick) | check model card | LLM judge + demo agent |

## Libraries

beautifulsoup4, numpy, pydantic, PyYAML, requests, scikit-learn, FastAPI, uvicorn, Streamlit, pytest,
httpx, and optionally torch, transformers, sentence-transformers, datasets. All are open source
(MIT / BSD / Apache-2.0).

## AI coding assistance

Parts of this codebase were drafted with AI assistants (Claude, GitHub Copilot) and then reviewed,
tested and modified by the team. Every member can explain the modules they own.
