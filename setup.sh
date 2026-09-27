#!/usr/bin/env bash
# One-command setup for macOS / Linux (judges run this).
#   ./setup.sh          core install (fast signals, gate, demo, dashboard) + tests
#   ./setup.sh --ml     also install the embedding model + injection classifier (~2 GB)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
$PY -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ needed"'

[ -d .venv ] || $PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e . -q --no-deps

if [[ "${1:-}" == "--ml" ]]; then
  pip install torch --index-url https://download.pytorch.org/whl/cpu -q
  pip install -r requirements-ml.txt -q
fi

[ -f .env ] || cp .env.example .env
python scripts/make_splits.py
python -m pytest -q

cat <<'MSG'

Setup complete. Try:
  source .venv/bin/activate
  python -m demo_agent.run_demo              # before/after in the terminal
  streamlit run dashboard/app.py             # the dashboard
  uvicorn api.main:app --port 8000           # the HTTP API (docs at /docs)
  python -m eval.run_eval                    # regenerate the numbers
For the real-LLM demo install Ollama (https://ollama.com), then:  ollama pull qwen2.5:3b
MSG
