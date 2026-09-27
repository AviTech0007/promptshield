"""Test setup: fast, deterministic, no internet, no LLM.

The heavy/optional signals (embeddings, classifier, LLM judge) are switched off so tests run
in seconds on any laptop and in GitHub Actions. They exercise the rules, TF-IDF similarity,
pre-processing, the taint tracker, the action gate, the audit log and the demo agent.
"""
import os
import sys
from pathlib import Path

os.environ["PS_DISABLE_CLASSIFIER"] = "1"
os.environ["PS_DISABLE_EMBEDDINGS"] = "1"
os.environ["PS_DISABLE_JUDGE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from promptshield import Shield  # noqa: E402


@pytest.fixture
def shield(tmp_path) -> Shield:
    return Shield(db_path=tmp_path / "test.db", use_judge=False)
