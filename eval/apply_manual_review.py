"""
One-time manual review pass over the 200 LLM-drafted golden labels 
(eval/golden_set.jsonl), applied by hand-reading every entry once.

Corrections made:
- ebf2dab23611789ce330b859ac5f0c85_0: relabeled account_access_or_security
    -> product_or_service_inquiry. The LLM label pass mistook a checkout/address-validation complaint for an account-security issue; the text has nothing to do with login/security.
- 43533f7de8e8eb500363e8bf91562350_4: DROPPED. Hindi-language text that
    slipped through the English-only langdetect filter in data_prep.py (short/code-mixed text is where langdetect is least reliable) - keeping it would violate this project's own documented English-only scope.

Flagged as low-confidence rather than "corrected", because the text is a short, out-of-context fragment (a mid-conversation follow-up picked up by single-turn exchange extraction, or a near-content-free tweet) where I don't have enough signal to be more confident than the LLM's guess:
27c2a3cfe26a20a47fd27667c5b6b580_0, ca5616e0d55a80cdfb786a45fedbb027_6, 6a89b73127f73a1c4c5431413718fe82_2, fbd92c7acb8a249be2699740f6aeafb0_0, 5d248f84b34cd687071d7162038b3d59_2, 22821db6992f5a78889b9da12ffdf442_7, ef3302fc8c09fc4bd6d33eca776ac1b2_0, bb2dbd906b21cdc3809703e4f3229698_3, 381caaf014aa15fcec901a41409558dc_5
seven of This is intentionally a small, targeted pass, not a full relabel these nine flags kept their original label because it was still the best available fit, just under-evidenced. See report/REPORT.md's "misleading headline number" section: any golden-set-derived accuracy number inherits this noise.

"""
import json
from pathlib import Path
EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "golden_set.jsonl"
RELABEL = {"ebf2dab23611789ce330b859ac5f0c85_0": "product_or_service_inquiry"}
DROP = {"43533f7de8e8eb500363e8bf91562350_4"}
LOW_CONFIDENCE_NOTE = "manual review: short/out-of-context fragment, label is a best guess"

LOW_CONFIDENCE_IDS = {"27c2a3cfe26a20a47fd27667c5b6b580_0",

"ca5616e0d55a80cdfb786a45fedbb027_6",
"6a89b73127f73a1c4c5431413718fe82_2",

"fbd92c7acb8a249be2699740f6aeafb0_0",

"5d248f84b34cd687071d7162038b3d59_2",

"22821db6992f5a78889b9da12ffdf442_7",
"ef3302fc8c09fc4bd6d33eca776ac1b2_0",

"bb2dbd906b21cdc3809703e4f3229698_3",

"381caaf014aa15fcec901a41409558dc_5",}

def main():
    with GOLDEN_PATH.open() as f:
        rows = [json.loads(line) for line in f]
    out = []
    for row in rows:
        eid = row["exchange_id"]
        if eid in DROP:
            continue
        if eid in RELABEL:
            row["gold_intent"] = RELABEL[eid]
            row["notes"] = (row.get("notes") or "") + " | manual review: relabeled, misclassified"
        if eid in LOW_CONFIDENCE_IDS:
            row["low_confidence"] = True
            row["notes"] = (row.get("notes") or "") + " | " + LOW_CONFIDENCE_NOTE
        out.append(row)
    with GOLDEN_PATH.open("w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(out)} examples after manual review (was {len(rows)})")



if __name__ == "__main__":
    main()
