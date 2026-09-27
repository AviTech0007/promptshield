"""The test harness: every number you show judges comes from here.

    python -m eval.run_eval                 # fast signals, simulated agent
    python -m eval.run_eval --judge         # also use the LLM judge (needs Ollama)
    python -m eval.run_eval --brain ollama  # attack-success rate with the real LLM agent

Measures, on the held-out TEST split only:
  1. Detection precision / recall / F1   (an item counts as "flagged" if level is warn or quarantine)
  2. False-alarm rate on tricky genuine emails (type == hard_negative)
  3. Ablation: each signal alone, each signal removed, and the existing classifier alone
  4. Recall on hidden-text variants (each test attack hidden inside HTML with display:none)
  5. Attack success rate of the demo agent: without shield, with full shield, with gate only
  6. Scan latency (median, p95)

Writes eval/results.json (read by the dashboard) and eval/results.md (paste into README).
REPORT WHAT YOU MEASURE, including weak numbers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demo_agent.agent import EmailAgent                    # noqa: E402
from demo_agent.brains import get_brain                    # noqa: E402
from demo_agent.mailstore import MailStore                 # noqa: E402
from demo_agent.run_demo import TASK                       # noqa: E402
from promptshield import Shield                            # noqa: E402
from promptshield.config import DATA_DIR                   # noqa: E402
from promptshield.signals import classifier, similarity    # noqa: E402

OUT_JSON = ROOT / "eval" / "results.json"
OUT_MD = ROOT / "eval" / "results.md"


def load_test() -> tuple[list[dict], str]:
    path = DATA_DIR / "splits" / "test.jsonl"
    if not path.exists():
        raise SystemExit("Run `python scripts/make_splits.py` first.")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows, str(path.relative_to(ROOT))


def make_shield(db: Path, judge: bool, **use) -> Shield:
    return Shield(db_path=db, use_judge=judge, **use)


def metrics(rows: list[dict], flagged: list[bool]) -> dict:
    tp = sum(1 for r, f in zip(rows, flagged) if f and r["label"] == 1)
    fp = sum(1 for r, f in zip(rows, flagged) if f and r["label"] == 0)
    fn = sum(1 for r, f in zip(rows, flagged) if not f and r["label"] == 1)
    tn = sum(1 for r, f in zip(rows, flagged) if not f and r["label"] == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hard = [f for r, f in zip(rows, flagged) if r.get("type") == "hard_negative"]
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "fpr_all_clean": round(fp / (fp + tn), 3) if fp + tn else 0.0,
            "fpr_hard_negatives": round(sum(hard) / len(hard), 3) if hard else 0.0,
            "n": len(rows), "n_hard_negatives": len(hard)}


def run_config(rows: list[dict], shield: Shield) -> tuple[dict, list[float]]:
    flagged, latencies = [], []
    for i, r in enumerate(rows):
        res = shield.scan(r["text"], source="eval", session_id=f"eval{i}")
        flagged.append(res.level != "allow")
        latencies.append(res.latency_ms)
        shield.reset_session(f"eval{i}")
    return metrics(rows, flagged), latencies


def attack_success(brain_name: str, db: Path, judge: bool, extra_attacks: list[str]) -> dict:
    """Run the agent on single-email inboxes. Only attacks that SUCCEED without the shield
    are counted, then we check how many still succeed with it."""
    brain = get_brain(brain_name)
    episodes: list[MailStore] = []
    base = MailStore.load()
    for e in base.inbox:
        if e.get("kind", "").startswith("attack"):
            episodes.append(MailStore(inbox=[dict(e)], user_email=base.user_email))
    for i, text in enumerate(extra_attacks):
        episodes.append(MailStore(inbox=[{"id": f"t{i}", "kind": "attack_test", "from": "someone@external.example",
                                          "subject": "Test attack", "body": text}], user_email=base.user_email))

    def leaked(store: MailStore, shield: Shield | None) -> bool:
        return bool(EmailAgent(store, brain, shield=shield).run(TASK).leaked)

    worked = [s for s in episodes if leaked(MailStore(inbox=[dict(x) for x in s.inbox], user_email=s.user_email), None)]
    if not worked:
        return {"episodes": len(episodes), "effective_without_shield": 0}
    full = sum(leaked(MailStore(inbox=[dict(x) for x in s.inbox], user_email=s.user_email),
                      make_shield(db, judge)) for s in worked)
    gate_only = sum(leaked(MailStore(inbox=[dict(x) for x in s.inbox], user_email=s.user_email),
                           make_shield(db, False, use_heuristics=False, use_similarity=False,
                                       use_classifier=False)) for s in worked)
    return {"episodes": len(episodes), "effective_without_shield": len(worked), "brain": brain_name,
            "without_shield": 1.0, "with_shield": round(full / len(worked), 3),
            "gate_only": round(gate_only / len(worked), 3)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--judge", action="store_true", help="use the LLM judge for unclear cases")
    p.add_argument("--brain", choices=["simulated", "ollama"], default="simulated")
    args = p.parse_args()

    rows, test_name = load_test()
    tmp = Path(tempfile.mkdtemp()) / "eval.db"
    cls_ok = classifier.is_available()
    print(f"Test set: {test_name} ({len(rows)} rows) | similarity={similarity.backend_name()} "
          f"| classifier={'on' if cls_ok else 'off'} | judge={'on' if args.judge else 'off'}")

    configs = {"Full system": dict(),
               "Rules only": dict(use_similarity=False, use_classifier=False),
               "Similarity only": dict(use_heuristics=False, use_classifier=False),
               "Without rules": dict(use_heuristics=False),
               "Without similarity": dict(use_similarity=False)}
    if cls_ok:
        configs["Classifier only (existing tool baseline)"] = dict(use_heuristics=False, use_similarity=False)
        configs["Without classifier"] = dict(use_classifier=False)

    ablation, full_metrics, full_lat = [], None, []
    for name, use in configs.items():
        judge = args.judge and name == "Full system"
        m, lat = run_config(rows, make_shield(tmp, judge, **use))
        ablation.append({"configuration": name, **{k: m[k] for k in ("precision", "recall", "f1",
                                                                      "fpr_hard_negatives")}})
        print(f"  {name:42s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} "
              f"FPR(hard)={m['fpr_hard_negatives']:.2f}")
        if name == "Full system":
            full_metrics, full_lat = m, lat

    # hidden-text variants of every test attack
    attacks = [r for r in rows if r["label"] == 1]
    hidden_rows = [{"text": f"<p>Hi, see the update below.</p><div style='display:none'>{r['text']}</div>",
                    "label": 1, "type": "hidden_variant"} for r in attacks]
    hm, _ = run_config(hidden_rows, make_shield(tmp, False))
    print(f"  Recall on hidden-text variants: {hm['recall']:.2f}")

    # attack success rate (only attacks with an address the agent can act on)
    with_addr = [r["text"] for r in attacks if "@" in r["text"]][:40]
    asr = attack_success(args.brain, tmp, args.judge, with_addr)
    print(f"  Attack success rate: {asr}")

    lat_sorted = sorted(full_lat)
    result = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "test_set": f"{test_name} ({len(rows)} rows)",
        "config": {"similarity": similarity.backend_name(), "classifier": cls_ok, "judge": args.judge},
        "full_system": full_metrics,
        "ablation": ablation,
        "hidden_variant_recall": hm["recall"],
        "attack_success_rate": asr if asr.get("effective_without_shield") else {},
        "latency_ms": {"median": round(statistics.median(full_lat), 1),
                       "p95": round(lat_sorted[int(0.95 * (len(lat_sorted) - 1))], 1)},
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    md = ["| Configuration | Precision | Recall | F1 | False alarms (tricky genuine) |",
          "|---|---|---|---|---|"]
    md += [f"| {a['configuration']} | {a['precision']:.2f} | {a['recall']:.2f} | {a['f1']:.2f} | "
           f"{a['fpr_hard_negatives']:.0%} |" for a in ablation]
    md.append("")
    md.append(f"Recall on hidden-text variants: **{hm['recall']:.0%}**  ")
    if result["attack_success_rate"]:
        a = result["attack_success_rate"]
        md.append(f"Attack success rate ({a['brain']} agent, {a['effective_without_shield']} attacks that worked "
                  f"without protection): **100% → {a['with_shield']:.0%}** with PromptShield, "
                  f"**{a['gate_only']:.0%}** with the action gate alone  ")
    md.append(f"Median scan time: **{result['latency_ms']['median']} ms** (p95 {result['latency_ms']['p95']} ms)  ")
    md.append(f"_Test set: {result['test_set']}; similarity: {result['config']['similarity']}; "
              f"classifier: {'on' if cls_ok else 'off'}; judge: {'on' if args.judge else 'off'}; "
              f"generated {result['generated_at']}_")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
