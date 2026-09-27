# PromptShield: implementation guide

This guide walks through **every file in the repo in the order you should build and understand it**, then lists the **later steps that tie up the demo** for the 28 Sep prototype submission and the 30 Sep – 1 Oct final.

Each file entry says: what it's for, how it works, who owns it, and how to check it works.

**Owners** (swap names in):

- **A: clean-up & rules.** `preprocess.py`, `heuristics.py`, `patterns.yaml`
- **B: models & numbers.** `similarity.py`, `classifier.py`, `fuse.py`, `train/`, `eval/`, `scripts/`
- **C: agent & gate.** `judge.py`, `taint.py`, `gate.py`, `policies/`, `audit.py`, `shield.py`, `demo_agent/`
- **D: product & story.** `dashboard/`, `api/`, `README.md`, `docs/`, deck, video

> **Rule for the whole team:** every member must be able to explain, without notes, every file they own. The rules allow judges to inspect commit history and ask any member about any file. AI tools helped write the first version; your understanding is what gets marked.

---

## Build order at a glance

Every file in the repo, in the order to build and understand it. The phase numbers match the sections below.

| # | File | Phase | Owner | One-line purpose |
|---|---|---|---|---|
| 1 | `requirements.txt`, `requirements-ml.txt`, `pyproject.toml` | 0 | all | Libraries; makes `promptshield` pip-installable |
| 2 | `setup.sh` / `setup.bat`, `.env.example`, `.gitignore`, `LICENSE`, `.github/workflows/tests.yml` | 0 | all | One-command setup, settings, open-source licence, CI |
| 3 | `promptshield/config.py` | 1 | B | Every tunable number in one place |
| 4 | `promptshield/types.py` | 1 | all | Shared data shapes (scan result, gate decision, ...) |
| 5 | `promptshield/preprocess.py` | 2 | A | Separates visible from hidden text, finds hiding tricks, chunks |
| 6 | `promptshield/signals/patterns.yaml` + `heuristics.py` | 2 | A | Signal 1: explainable rules |
| 7 | `promptshield/signals/similarity.py` | 2 | B | Signal 2: similarity to known attacks |
| 8 | `promptshield/signals/classifier.py` | 2 | B | Signal 3: existing open-source injection classifier |
| 9 | `promptshield/signals/judge.py` | 2 | C | Signal 4: LLM judge for unclear sentences only |
| 10 | `promptshield/fuse.py` | 2 | B | Combines signals into a 0–100 risk and a reason |
| 11 | `promptshield/taint.py` | 3 | C | Remembers which values came from untrusted content |
| 12 | `policies/email_agent.yaml` | 3 | C | Which tools are risky, which arguments to check |
| 13 | `promptshield/gate.py` | 3 | C | Checkpoint 2: allow / ask a human / block |
| 14 | `promptshield/audit.py` | 3 | C/D | SQLite audit log + approval queue |
| 15 | `promptshield/shield.py`, `promptshield/__init__.py` | 3 | C | The public SDK that wires everything together |
| 16 | `demo_agent/inbox.json` | 4 | C | 10 mock emails: 4 attacks, 2 tricky genuine, 4 normal |
| 17 | `demo_agent/mailstore.py` | 4 | C | Mock inbox/outbox (never sends real email) |
| 18 | `demo_agent/brains.py` | 4 | C | Real Ollama LLM or a labelled simulated agent |
| 19 | `demo_agent/agent.py`, `run_demo.py` | 4 | C | The email agent + terminal before/after demo |
| 20 | `api/main.py` | 5 | D | HTTP API for agents in any language |
| 21 | `dashboard/app.py` | 5 | D | The Streamlit demo, approvals, try-to-break-it, audit, numbers |
| 22 | `data/*` + `scripts/download_datasets.py`, `import_bipia.py`, `make_splits.py` | 6 | B + all | Datasets and honest train/val/test splits |
| 23 | `train/train_fusion.py` | 6 | B | Learns how much to trust each signal |
| 24 | `eval/run_eval.py` | 6 | B | Every number you show judges |
| 25 | `tests/*` | 7 | all | 60 tests; run before every push |
| 26 | `scripts/preflight.py` | 8 | D | Demo-day readiness check |
| 27 | `README.md`, `docs/DEMO_SCRIPT.md`, `docs/JUDGE_QA.md`, `data/LICENSES.md` | 8 | D | Submission docs, demo script, judge prep, disclosure |

---

## What already works (measured in this build)

- `pytest`: **60 tests pass** in under 2 seconds. 12 of them exercise the optional ML/LLM code paths with fake stand-in models (`tests/test_optional_backends.py`).
- `python scripts/preflight.py` runs cleanly (0 failures; the warnings are the to-do items in Phase 8).
- API smoke test: `/health`, `/scan` (hidden attack → quarantine) and `/check_action` (attacker address → block) all respond correctly.
- Terminal demo (`python -m demo_agent.run_demo`, simulated brain):
  - Without the shield, 17 emails leak to 4 attacker addresses.
  - With the shield, nothing leaks, and the genuine "forward to HR" request still goes through.
  - With **detectors off**, the gate alone still holds the rephrased attack (e7).
- Dashboard: every tab loads, the runs work, and approvals execute.
- Evaluation on the **tiny starter test split (16 rows)**: F1 0.93, 0 false alarms on 4 tricky genuine emails, attack success 100% → 0%, median scan about 1 ms.

What has **not** been tested with the real thing yet (the build machine couldn't reach Hugging Face or Ollama):

- the real ML models (embeddings, classifier), the real LLM judge and the real Ollama brain. Their code paths *are* tested with fakes, so what's left is download, speed and model behaviour;
- datasets downloaded from Hugging Face.

The code falls back gracefully when those pieces are missing. You must run them on your own laptops (Phase 8).

**The seed numbers above are not your results.** The rules were tuned while looking at the same seed set, and 16 rows is far too few. Replace them with numbers from the full, larger test set (step 8.3) before you put any number on a slide.

---

## Phase 0: Setup and project files (everyone, first hour)

| File | What it is for |
|---|---|
| `requirements.txt` | The core libraries, pinned to the versions this project was tested with. Enough for everything except the two ML signals. |
| `requirements-ml.txt` | Optional heavy libraries: torch, transformers, sentence-transformers, datasets. They switch on the embedding and classifier signals. Install the **CPU** build of torch (the command is inside the file) to save about 2 GB. |
| `pyproject.toml` | Makes `promptshield` a real pip-installable package (`pip install -e .`). This is what backs the "drop-in SDK" claim on your slides. Also configures pytest. |
| `setup.sh` / `setup.bat` | The one-command setup judges will run. It creates `.venv`, installs everything, copies `.env`, builds the data splits and runs the tests. |
| `.env.example` → `.env` | Settings you may want to change without touching code: Ollama URL and model, classifier model, a Groq key, and switches to turn signals off on slow laptops. |
| `.gitignore` | Keeps `runtime/` (SQLite files, caches), `.venv/` and downloaded datasets out of git. |
| `LICENSE` | MIT. The rules require an open-source licence. |
| `.github/workflows/tests.yml` | GitHub Actions runs the tests on every push. The green badge in the README is quick visual proof of stability. |

**Do now:**

1. Install Python 3.11 and git.
2. The team lead creates an **empty public** GitHub repo called `promptshield` (no README, no licence; they're already here), then pushes this code:

   ```bash
   cd promptshield
   git init -b main
   git add .
   git commit -m "Initial scaffold (AI-assisted, see README disclosure)"
   git remote add origin https://github.com/<your-username>/promptshield.git
   git push -u origin main
   ```

3. Add the other three members as collaborators (repo Settings → Collaborators).
4. Each member clones it, runs `./setup.sh` (Windows: `setup.bat`) and gets `60 passed`.
5. Run `python -m demo_agent.run_demo` and read the output together.
6. Run `python scripts/preflight.py`. Every warning it prints is a Phase 8 to-do.
7. Install Ollama from ollama.com, then run `ollama pull qwen2.5:3b`.
8. Replace `<your-org>` in the README badge with your GitHub username.

From now on, each person commits their own changes from their own account (`git pull`, edit, `python -m pytest -q`, `git add`, `git commit`, `git push`).

---

## Phase 1: Shared foundations

### `promptshield/config.py` (B)

This file holds every tunable number:

- signal weights;
- the "unclear band" for sending sentences to the LLM judge;
- the warn and quarantine thresholds (40 and 70);
- chunk size;
- model names.

It also reads `.env`. **Why it matters:** when you tune something, you change it here, in one place, and the change is easy to explain.

### `promptshield/types.py` (everyone reads)

This file defines the data shapes every module passes around, as pydantic models:

- `Finding`: a hiding trick that was found
- `Chunk`: one sentence of the content
- `SignalScore`: one signal's verdict on one chunk
- `ChunkResult`, `ScanResult`: what `shield.scan()` returns
- `GateDecision`: what the action gate returns

Pydantic validates these shapes and gives FastAPI its API docs for free.

---

## Phase 2: Checkpoint 1, scanning content going in

The order below matches the pipeline.

### 1. `promptshield/preprocess.py` (A)

Turns raw content (HTML or text) into **visible text**, **hidden text**, **findings** and **chunks**.

- **HTML:** BeautifulSoup walks every text node and checks it and its parents for inline styles that hide text:
  - `display:none`, `visibility:hidden`, `opacity:0`;
  - font size 0 or 1;
  - white text colour;
  - positioned off-screen.

  HTML comments, long `alt`/`title` attributes and `<meta>` content count as hidden too.
- **Unicode:**
  - decodes invisible *tag characters* ("ASCII smuggling") back into readable text;
  - removes zero-width and text-direction characters;
  - flags words that mix Latin with Cyrillic or Greek look-alike letters, then folds those letters back to Latin so the rules still match;
  - applies NFKC normalisation.
- **Encoded content:** decodes base64 blobs that turn out to be readable text, and flags markdown images whose URL carries data.
- **Chunking:** splits into sentences of at most 400 characters (about 100 tokens). The classifier models only read 512 tokens, so a long email must never be silently cut off.

The key design decision is that `clean_text` (what the agent gets) contains **only visible text**. Hidden text is scored but never shown to the agent.

**Verify:** `pytest tests/test_preprocess.py`

**Known limit (put it in the README):** only *inline* styles are understood.

### 2. `promptshield/signals/patterns.yaml` and `heuristics.py` (A), Signal 1: rules

- **What a rule is:** a regex with a weight and a plain-English reason, for example "Tries to cancel the agent's existing instructions".
- **How matches combine (noisy-OR):** `1 − (1−w1)(1−w2)…`. Several weak hits add up, but no single one takes over.
- **Context rules** (`standalone: false`, e.g. "urgent", "don't tell anyone") count half when they match alone. Genuine emails use those words all the time.
- **Hidden text:** text that humans can't see scores at least 0.35. Hidden text that also matches a rule scores at least 0.9.

**Adding a rule:** edit the YAML, then add one attack line and one genuine line to `tests/test_heuristics.py`.

**Verify:** `pytest tests/test_heuristics.py`

**What to say to judges:** rules make decisions explainable. They are not the main defence.

### 3. `promptshield/signals/similarity.py` (B), Signal 2: similarity to known attacks

For each chunk it finds the 5 nearest labelled examples in the **TRAIN split** and computes:

`score = (share of attacks among those neighbours) × (how close the nearest one is)`

Text that isn't like anything in the library therefore scores low, instead of landing at 0.5.

**Backends:**

- **Embeddings** (all-MiniLM-L6-v2) if `requirements-ml.txt` is installed. They catch reworded attacks.
- **TF-IDF character n-grams** otherwise. This backend is always available.

Embeddings are cached in `runtime/`.

**Verify:** the dashboard sidebar shows "Similarity backend: embeddings" after `./setup.sh --ml`.

### 4. `promptshield/signals/classifier.py` (B), Signal 3: an existing injection classifier

This runs `protectai/deberta-v3-base-prompt-injection-v2` by default. The alternative is Llama Prompt Guard 2:

1. accept Meta's licence on its Hugging Face page;
2. run `huggingface-cli login`;
3. set `PS_CLASSIFIER_MODEL` in `.env`.

The signal outputs P(injection) per chunk. If the model can't load, it reports "unavailable" and `fuse.py` re-balances the weights, so the app never crashes.

This signal is also the **baseline** in the ablation study. Showing that the full system beats the classifier alone answers "isn't this just a wrapper?".

### 5. `promptshield/signals/judge.py` (C), Signal 4: LLM judge, unclear cases only

- **When it runs:** only when a chunk's fast score falls between 0.30 and 0.70.
- **What it's asked:** one narrow question, with a fixed JSON answer format.
- **Backends:** Ollama first, then Groq if you set a key. Only ever send synthetic test data to Groq.

The judge can be attacked too, so it is hardened:

- the content is wrapped in `<data>` tags and described as data, not instructions;
- the output is validated with pydantic;
- an invalid output scores **0.75**, which counts as suspicious, never as safe;
- it can only move the score halfway (`judge_weight`);
- the gate never trusts it alone.

### 6. `promptshield/fuse.py` (B), combining signals into one risk score

- **Per chunk:** a weighted average of the available fast signals.
  - If a single signal is 0.9 or higher, it isn't averaged away (`strong_signal_floor`).
  - If the score falls in the unclear band, the LLM judge adjusts it.
- **Per document:** risk is the highest chunk score, raised to a minimum when a hiding trick was found. For example, Unicode tag characters set a minimum of 45.
- **Levels:** allow below 40, warn from 40 to 69, quarantine at 70 and above.
- **Learned weights:** if `train/train_fusion.py` has produced `runtime/fusion_weights.json` from **at least 100** validation examples, a logistic regression replaces the hand weights. It's normalised so that "no evidence" scores 0. With fewer examples the learned weights are ignored, because they'd be noise.
- `explain()` writes the plain-English reason shown to users.

---

## Phase 3: Checkpoint 2, holding risky actions

### 7. `promptshield/taint.py` (C), remembering what came from outside

Every scanned piece of content becomes a **taint record**. It stores:

- where the content came from and its risk level;
- the values an attacker would want the agent to use: email addresses and their domains, URLs and hosts, long numbers.

Obfuscation is undone first, so "billing dash verify at mailbox dot example" becomes `billing-verify@mailbox.example`.

The **trusted** values are:

- anything in the user's own request (`shield.set_user_request()`);
- the policy's contact list.

`origin_of(value)` answers: "did this value come from untrusted content, and did the user never mention it?"

### 8. `policies/email_agent.yaml` (C)

- **Tool risk levels:** every tool is `low`, `medium`, `high` or `critical`.
- **`watch_args`:** destination arguments that get the taint check.
- **`bulk_args`:** arguments that make an action affect many items.
- **`trusted_values`:** the demo company's contacts.

Unknown tools default to **high**. Customise this file for any other agent.

### 9. `promptshield/gate.py` (C), the action gate

`decide(tool, args, session)` returns allow, require_approval or block:

| Tool risk | Decision |
|---|---|
| **low** | always allow (still logged) |
| **medium** | ask a human if a destination argument is tainted |
| **high** | **block** if the tainted value came from quarantined content; **ask a human** if it's tainted, or suspicious (warn-level) content reached the agent, or it's a bulk action; otherwise allow |
| **critical** | always ask a human (block if the value came from quarantined content) |

This is why a rephrased attack that slips past every detector still can't forward mail to the attacker.

**Verify:** `pytest tests/test_gate.py`

### 10. `promptshield/audit.py` (C/D)

SQLite at `runtime/promptshield.db`, with two tables:

- `scans`: every check and its result.
- `actions`: every tool call, plus the status of its approval: `allowed`, `pending`, `approved`, `denied`, `executed` or `blocked`.

This is the audit log and approval queue that the dashboard and API show.

### 11. `promptshield/shield.py` (C owns; everyone must understand it)

The public API that ties everything together:

- **`scan()`:** runs preprocess → signals → fuse, registers the content as tainted, logs it, and returns a `ScanResult`.
- **`check_action()`:** runs the gate and logs the decision.
- **`@guard_tool()`:** wraps a tool function so every call goes through the gate first. A held call raises `ActionHeld`.
- **`approve()` / `deny()`:** the human decision. Approving actually runs the stored tool.
- **`check_output()`:** scans the agent's own reply for links that would leak data.

`promptshield/__init__.py` exports `Shield` and `ActionHeld`.

---

## Phase 4: The demo agent (C)

### 12. `demo_agent/inbox.json`

Ten mock emails:

- **4 attacks:**
  - e3: hidden white-on-white text
  - e5: an HTML comment
  - e7: a *visible, polite, reworded* request
  - e9: invisible Unicode tag characters
- **2 tricky genuine emails:**
  - e4: "forward this to HR"
  - e8: "ignore my previous email"
- **4 normal emails.**

`kind` is only used for evaluation and badges; the agent never sees it. All addresses end in `.example`.

### 13. `demo_agent/mailstore.py`

The mock inbox and outbox. **No real email is ever sent.**

`llm_view()` imitates how a typical unprotected pipeline turns HTML into text for an LLM: it keeps hidden text and comments. That's why the attacks work "before". `leaked_to()` counts outbox items sent to untrusted addresses.

### 14. `demo_agent/brains.py`

- **`OllamaBrain`:** a real local LLM with a normal, non-rigged prompt and JSON output. Use it for the live demo **if it actually falls for the attacks on your laptop** (step 8.1).
- **`SimulatedBrain`:** a deterministic stand-in for a completely gullible agent. It obeys any "forward/send/pass along … to address" it reads. Tests and the evaluation use it so results can be reproduced. **Always call it "simulated" on stage.**

### 15. `demo_agent/agent.py`

The loop for each email:

1. read the email;
2. **[scan]**;
3. the brain decides a summary and actions;
4. for each action, **[gate]**, then the tool runs.

Without a shield it reads `llm_view()` and runs every action immediately.

### 16. `demo_agent/run_demo.py`

The terminal before/after demo. Flags:

- `--brain ollama` uses the real LLM;
- `--emails e3,e7` runs only some emails;
- `--mode shielded` runs only the protected version;
- `--detectors off` proves the gate works alone;
- `--no-judge` skips the LLM judge.

---

## Phase 5: Interfaces (D)

### 17. `api/main.py`

FastAPI, so agents written in any language can use PromptShield. Start it with `uvicorn api.main:app --port 8000` and open `/docs` for interactive documentation, which is nice to show judges.

Endpoints:

- `/scan`
- `/check_action`
- `/session/user_request`
- `/actions/pending`
- `/actions/{id}/approve` and `/actions/{id}/deny`
- `/audit/*`

### 18. `dashboard/app.py`

The Streamlit dashboard. Run it with `streamlit run dashboard/app.py`.

**Sidebar:**

- choose the brain;
- **detectors on/off toggle** (for the defence-in-depth moment);
- LLM judge toggle;
- which backends are loaded;
- reset.

**Tabs:**

1. **Live demo:** without and with the shield, side by side. Shows risk badges, the four signal bars, highlighted sentences (hidden ones marked 🙈) and each action's outcome.
2. **Approvals:** held actions with their reasons, and Approve & run / Deny buttons.
3. **Try to break it:** judges paste their own attack, see every signal, then test the gate on it.
4. **Audit log.**
5. **Evaluation:** reads `eval/results.json`.

---

## Phase 6: Data and numbers (B, with everyone writing examples)

| File | Purpose |
|---|---|
| `data/seed_corpus.jsonl` | 40 attacks + 40 clean starter examples (21 of them tricky genuine emails). Written with AI help, so disclose it. |
| `data/team_corpus.jsonl` | **Empty. You fill it:** about 40 attacks + 40 clean per member. |
| `data/team_test_blind.jsonl` | **You create it:** one member writes 30+ attacks and 30+ clean examples *without reading* `patterns.yaml`. These always go to TEST. They are what makes your numbers believable. |
| `data/README.md` | The row format and how to write good examples. |
| `data/LICENSES.md` | Disclosure table for datasets, models, libraries and AI assistance. **Verify every licence** before submitting. |
| `scripts/download_datasets.py` | Downloads deepset and ProtectAI data from Hugging Face into `data/raw/`. |
| `scripts/import_bipia.py` | After you `git clone` Microsoft BIPIA into `data/raw/BIPIA`, wraps its attacks in email-style text. |
| `scripts/make_splits.py` | Builds 60/20/20 train/val/test splits, stratified and de-duplicated across splits. The blind file always goes to test. |
| `train/train_fusion.py` | Logistic regression on the **VAL** split learns the signal weights and saves `runtime/fusion_weights.json`. It uses VAL, not TRAIN, because similarity already uses TRAIN as its library. |
| `eval/run_eval.py` | **Every number you show.** See below. |

What `eval/run_eval.py` measures on the TEST split:

- precision, recall and F1;
- false-alarm rate on tricky genuine emails;
- an ablation: each signal alone, each removed, and the classifier alone as the baseline;
- recall on hidden-HTML versions of every test attack;
- attack success rate without the shield, with the full shield, and with the gate only;
- latency.

It writes `eval/results.json` (read by the dashboard) and `eval/results.md` (paste into the README).

## Phase 7: Tests

- `tests/conftest.py`: turns off the heavy signals so tests are fast and need no internet.
- `test_preprocess.py`, `test_heuristics.py`, `test_gate.py`: one module each (hidden text, rules, taint + gate).
- `test_end_to_end.py`: the whole story (agent leaks → shield stops it → gate alone stops it → approval executes → API works).
- `test_optional_backends.py`: the classifier, embeddings, LLM judge and Ollama brain, each with a small **fake** model. They check label mapping, caching, "invalid judge output counts as suspicious" and JSON parsing, without downloading anything.
- Run `python -m pytest -q` before every push. GitHub Actions runs it too.

### `scripts/preflight.py` (D)

A demo-day readiness check. `python scripts/preflight.py` takes a few seconds. `--full` also loads the ML models, asks Ollama whether the real model falls for the attacks (step 8.1), and runs the test suite.

- **OK:** ready.
- **WARN:** the demo works, but fix it before judging (small test set, README placeholders, missing git tag, ...).
- **FAIL:** fix now.

Run it after setup, the night before the final, and 30 minutes before you present.

---

## Phase 8: Later steps to tie up the demo

Work through these in order. Steps 8.1–8.8 are for **today and tomorrow (27–28 Sep)**. Step 8.9 is the event.

### 8.1 Check the real LLM actually falls for the attack (C, today, most important)

1. Run `ollama serve`, then `ollama pull qwen2.5:3b`.
2. Run `python -m demo_agent.run_demo --brain ollama --emails e3,e7,e9 --mode unshielded`.
3. **If it leaks**, you have a real "before". Also run `--mode shielded` to confirm nothing leaks.
4. **If it doesn't leak:**
   - try `llama3.2:3b`, `qwen2.5:7b`, `gemma2:2b` or `mistral` (set `PS_OLLAMA_MODEL` in `.env`);
   - try each attack email on its own;
   - small models behave differently, so keep whichever reliably complies.
5. Record the model name that works in the README.
6. **If none comply reliably:** use the simulated brain for the "before" and say so plainly: "this is a simulated gullible agent; real models comply some of the time, as our eval shows." Never pass the simulation off as a real model.
7. Screen-record a successful real-LLM "before" as a backup clip.

### 8.2 Turn on the ML signals (B, today)

1. Run `./setup.sh --ml` (or `setup.bat --ml`).
2. Open the dashboard. The sidebar should show "embeddings" and "classifier: loaded".
3. The first run downloads the models, so do it on good Wi-Fi, never on stage.
4. If a laptop is too slow, set `PS_DISABLE_CLASSIFIER=1` in its `.env`.

### 8.3 Grow the data and produce real numbers (everyone writes, B runs)

1. Each member writes about 40 attacks and 40 clean examples (at least 15 tricky genuine) into `data/team_corpus.jsonl`.
2. One member writes `data/team_test_blind.jsonl` without looking at the rules.
3. Run `python scripts/download_datasets.py`.
4. Clone BIPIA and run `python scripts/import_bipia.py`.
5. Run `python scripts/make_splits.py`, then `python -m train.train_fusion`, then `python -m eval.run_eval`.
6. Paste `eval/results.md` into the README's Results section.
7. Update slide 6 with **measured** numbers next to your targets. If you missed a target, say so.

### 8.4 Tune on VAL only

If there are too many false alarms:

- adjust rule weights, `standalone` flags or the thresholds in `config.py`;
- check the effect on the **VAL** split;
- only then re-run the TEST evaluation, once.

Never tune while watching TEST numbers. That's how numbers stop being believable.

### 8.5 Dashboard and demo rehearsal (D)

- Do a full rehearsal run with the dashboard on the laptop you'll present from.
- Reset the demo from the sidebar before each rehearsal.
- Pre-run `eval.run_eval` so the Evaluation tab is filled in.
- Run `python scripts/preflight.py --full` on that laptop and clear every FAIL.

### 8.6 README, disclosure, tag and push (D)

1. Fill in the Team section, the Results section and the Ollama model you chose.
2. Verify every licence in `data/LICENSES.md`.
3. Commit, then run `git tag v0.9-pre-event` and `git push --tags`.

The tag is the line between "before the event" and "during the event".

### 8.7 Record the 2-minute walkthrough video (D)

- Follow `docs/DEMO_SCRIPT.md`.
- Use OBS or the Windows Game Bar. Use 1080p and increase the terminal and browser font size.
- Upload it unlisted to YouTube or Drive and link it in the README.

### 8.8 Final idea submission, 28 Sep

Submit:

- the repo link;
- the video;
- the prototype description. Use the "What already works" list at the top of this guide, updated with your real numbers.

### 8.9 The 24-hour event (30 Sep 11:00 → 1 Oct 11:59)

| Hours | Plan |
|---|---|
| 0–3 | Fix whatever mentors flagged. Add more tricky genuine emails if false alarms are high. |
| 3–10 | **One** stretch goal only: (a) an **MCP tool proxy** that applies `guard_tool` to any MCP server's tools; (b) a **second demo agent** (web-page summariser) to prove the SDK works for more than email; or (c) wire `check_output()` into the agent's final reply. |
| 10–15 | Retrain the weights and re-run the evaluation. Fix false alarms using VAL only. |
| 15–19 | README (list the "during the event" items), architecture image, final licence check. |
| 19–22 | Re-record the walkthrough video with the final build. |
| 22–24 | Rehearse twice. Submit on cyreneai.com well before 11:59. |

**Demo-day checklist:**

- `python scripts/preflight.py --full` shows 0 failures on the presenting laptop.
- Models are pre-pulled and caches are warm, so the demo works with no Wi-Fi.
- The laptop is charged and the dashboard is open.
- The backup video is on the desktop.
- `eval/results.json` is present.

### 8.10 Commit honestly

- Commit small and often, from **your own** GitHub account, with real timestamps. Never rewrite history.
- The first commit can be "Initial scaffold (AI-assisted), see README disclosure". That's allowed as long as it's disclosed.
- Your own commits after that show your work.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `classifier unavailable` message | Install `requirements-ml.txt`. For Prompt Guard, accept the licence and run `huggingface-cli login`. Or keep ProtectAI. |
| Similarity says `tfidf` | sentence-transformers isn't installed, or `PS_DISABLE_EMBEDDINGS=1` is set. |
| Ollama brain errors or times out | Run `ollama serve`, check `PS_OLLAMA_MODEL` is pulled, or try a smaller model. |
| Clean emails get "warn" | Retrain with at least 100 VAL examples, or delete `runtime/fusion_weights.json`. Check which rule fires with `heuristics.matched_rule_ids(text)`. |
| Dashboard shows old approvals | Click "Reset demo" in the sidebar. |
| Windows: `source` not found | Use `.venv\Scripts\activate`. |
