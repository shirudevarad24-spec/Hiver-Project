"""
Automated metrics: classifier accuracy/F1 for each of the three classifiers
against eval/golden_set.jsonl, and escalation-decision accuracy against the
same golden set's gold_escalate labels.
"""

import json
import sys
from pathlib import Path

from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classify import SimpleClassifier, TrivialClassifier, llm_classify  # noqa: E402
from draft import draft_reply  # noqa: E402
from retrieve import ExchangeRetriever  # noqa: E402
from route import ALWAYS_ESCALATE_INTENTS, SAFETY_PATTERNS, decide  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EVAL_DIR.parent / "data" / "processed"


def load_golden() -> list[dict]:
    with (EVAL_DIR / "golden_set.jsonl").open() as f:
        return [json.loads(l) for l in f]


def load_bootstrap_labels() -> list[dict]:
    with (PROCESSED_DIR / "bootstrap_labels.jsonl").open() as f:
        return [json.loads(l) for l in f]


def evaluate_classifiers(golden: list[dict], bootstrap: list[dict]) -> dict:
    train_labels = [r["intent"] for r in bootstrap]
    trivial = TrivialClassifier(train_labels)
    simple = SimpleClassifier([r["customer_text"] for r in bootstrap], train_labels)

    y_true = [g["gold_intent"] for g in golden]
    preds = {"trivial": [], "simple": [], "llm": []}
    for g in golden:
        preds["trivial"].append(trivial.predict(g["customer_text"]))
        preds["simple"].append(simple.predict(g["customer_text"]))
        preds["llm"].append(llm_classify(g["customer_text"])["intent"])
        print(f"  classified {len(preds['llm'])}/{len(golden)}", end="\r")
    print()

    results = {}
    for name, y_pred in preds.items():
        results[name] = {
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true),
            "report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
        }
    return results, preds


def evaluate_routing(golden: list[dict]) -> dict:
    # Exclude golden examples from the retrieval corpus - otherwise a golden
    # example's own (customer_text, support_text) pair can be retrieved as
    # its own grounding, trivially inflating similarity/groundedness.
    golden_ids = {g["exchange_id"] for g in golden}
    retriever = ExchangeRetriever.from_jsonl(exclude_ids=golden_ids)
    correct, rows = 0, []
    for g in golden:
        # Rule-only escalations don't need a draft at all - skip the LLM
        # drafting call in that case, mirroring decide()'s own short-circuit
        # and saving ~a third of the LLM calls this eval would otherwise cost.
        if g["gold_intent"] in ALWAYS_ESCALATE_INTENTS or SAFETY_PATTERNS.search(g["customer_text"]):
            draft = {"reply": "", "grounding": [], "grounded": False}
        else:
            draft = draft_reply(g["customer_text"], retriever)
        # Use a fixed high confidence here: this metric isolates the ROUTING
        # rules/judge, not the classifier's own confidence calibration.
        decision = decide(g["customer_text"], g["gold_intent"], 0.9, draft)
        predicted_escalate = decision["action"] == "escalate"
        correct += int(predicted_escalate == g["gold_escalate"])
        rows.append({
            "exchange_id": g["exchange_id"],
            "gold_escalate": g["gold_escalate"],
            "predicted_escalate": predicted_escalate,
            "reason": decision["reason"],
        })
        print(f"  routed {len(rows)}/{len(golden)}", end="\r")
    print()
    return {"accuracy": correct / len(golden), "rows": rows}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="limit number of golden examples to evaluate")
    args = parser.parse_args()

    golden = load_golden()
    if args.limit:
        golden = golden[: args.limit]
    bootstrap = load_bootstrap_labels()

    print(f"Evaluating classifiers on {len(golden)} golden examples...")
    clf_results, preds = evaluate_classifiers(golden, bootstrap)
    for name, r in clf_results.items():
        print(f"  {name:8s} accuracy={r['accuracy']:.3f} macro_f1={r['macro_f1']:.3f}")

    print(f"\nEvaluating routing on {len(golden)} golden examples...")
    routing_results = evaluate_routing(golden)
    print(f"  routing accuracy={routing_results['accuracy']:.3f}")

    out = {"classifiers": clf_results, "routing": {"accuracy": routing_results["accuracy"]}}
    out_path = EVAL_DIR / "metrics_results.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")