# Manual annotation guide

`manual_annotation_template.jsonl` contains 200 deterministically sampled
AmazonHelp exchanges: 180 random examples and 20 deliberately difficult ones.
It is an annotation packet, not an evaluation set yet.

For every row, fill in:

- `gold_intent`: exactly one intent name from `src/taxonomy.py`.
- `gold_escalate`: `true` if an automated reply should be reviewed by a human;
  otherwise `false`.
- `gold_escalate_reason`: a short reason, particularly for high-risk or
  ambiguous cases.
- `annotator_notes`: optional explanation of ambiguity or data-quality issues.

Review customer text independently. `support_text_actual` is supplied only as
historical context and should not determine the intent label by itself. Remove
examples that are non-English, unreadable, duplicates, or cannot be labeled
with reasonable confidence; replace them using the same sampling procedure and
record the replacement in the final report.

After all labels are filled, validate that `gold_escalate` values are JSON
booleans—not strings—and save the completed file as `eval/golden_set.jsonl`.
