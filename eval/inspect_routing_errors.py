"""Re-runs routing evaluation and dumps every mismatch (predicted vs. gold
escalation) with the stated reason, for the report's failure-analysis section."""

import json
from pathlib import Path

from metrics import evaluate_routing, load_golden

EVAL_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    golden = load_golden()
    results = evaluate_routing(golden)
    print(f"routing accuracy={results['accuracy']:.3f}")

    mismatches = [r for r in results["rows"] if r["gold_escalate"] != r["predicted_escalate"]]
    out_path = EVAL_DIR / "routing_mismatches.jsonl"
    id_to_text = {g["exchange_id"]: g["customer_text"] for g in golden}
    with out_path.open("w") as f:
        for m in mismatches:
            f.write(json.dumps({**m, "customer_text": id_to_text[m["exchange_id"]]}) + "\n")
    print(f"Wrote {len(mismatches)} mismatches -> {out_path}")
