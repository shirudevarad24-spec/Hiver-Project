# AmazonHelp AI Customer Support Agent — Hiver Take-Home Assignment

An AI-powered Tier-1 support agent for **AmazonHelp** Twitter customer support:
- **Classifies** incoming customer tweets into a focused 9-class taxonomy (Few-Shot LLM vs. TF-IDF + Logistic Regression vs. Trivial baseline).
- **Retrieves** historically resolved similar exchanges using TF-IDF vector search.
- **Drafts** grounded, brand-appropriate replies reusing official help links and resolution patterns.
- **Routes** whether to auto-reply or escalate to a human agent with stated deterministic risk reasons.

The comprehensive systems report, problem framing, baseline analysis, failure modes, and trade-offs are documented in:
- **[`report/REPORT.md`](report/REPORT.md)** (Full analysis and benchmark report)
- **[`report/decision_log.md`](report/decision_log.md)** (Key architectural decisions and rationale)

---

## Quickstart & Fast Verification (< 1 minute)

The repository comes with pre-computed, hand-reviewed evaluation benchmarks so you can verify the results immediately without making hundreds of API calls:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Inspect computed headline benchmark numbers
cat eval/metrics_results.json        # Accuracy, Macro-F1 across 3 baselines + routing
cat eval/agreement_results.json      # LLM-as-a-judge vs. Human Pearson agreement (r > 0.87)

# 3. Test the live end-to-end pipeline on an ad-hoc message
python src/pipeline.py --text "My package says delivered but never arrived"

# 4. Re-run metrics evaluation over the golden set
python eval/metrics.py
```

---

## Setup & Configuration

### Prerequisites
- Python 3.11+
- Virtual environment (`.venv`)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Provider Credentials
The system supports GroqCloud (default), xAI Grok, and Anthropic. A `.env` file in the project root is automatically loaded:

```bash
# Groq (Recommended - fast LPU inference)
export LLM_PROVIDER=groq
export GROQ_API_KEY="your-groq-key"
export GROQ_MODEL="qwen/qwen3.8-27b"

# Alternatively: xAI Grok
# export LLM_PROVIDER=grok
# export GROK_API_KEY="your-xai-key"

# Alternatively: Anthropic
# export LLM_PROVIDER=anthropic
# export ANTHROPIC_API_KEY="your-anthropic-key"
```

---

## Reproduce the Full Pipeline from Scratch

To regenerate all datasets, weak supervision labels, golden benchmark, and evaluation scores:

```bash
# 1. Data preparation & validation
python src/data_prep.py --brand AmazonHelp --limit 5000
python eval/validate_data.py

# 2. Generate weak supervision bootstrap labels
python src/bootstrap_labels.py -n 300

# 3. Build & review golden set
python eval/build_golden_set.py
python eval/apply_manual_review.py

# 4. Run metrics evaluation
python eval/metrics.py

# 5. Run LLM judge evaluation & routing error dump
python eval/run_judge_eval.py --n 10
python eval/inspect_routing_errors.py
```

---

## Repository Structure

```
├── README.md                           # Main project documentation
├── requirements.txt                    # Project dependencies
├── .env                                # Local API keys (gitignored)
├── data/
│   ├── raw/hf_conversations.parquet    # Cached raw Twitter support data
│   └── processed/                      # Cleaned exchanges, threads, and bootstrap labels
├── eval/
│   ├── golden_set.jsonl                # Golden evaluation benchmark (200 hand-labelled examples: 180 stratified + 20 edge cases)
│   ├── metrics_results.json            # Classifier & routing benchmark scores
│   ├── judged_replies.jsonl            # LLM-judge quality scores (1-5 rubric)
│   ├── human_scores.jsonl              # Human annotator scores for judge calibration
│   ├── agreement_results.json          # Pearson correlation (Judge vs Human)
│   ├── routing_mismatches.jsonl        # Failure analysis mismatch dump
│   └── ANNOTATION_GUIDE.md             # Golden set sampling & manual review guide
├── report/
│   ├── REPORT.md                       # Comprehensive evaluation & systems report
│   ├── decision_log.md                 # Architecture decisions & trade-offs
│   └── data_summary.json               # Dataset split statistics
└── src/
    ├── pipeline.py                     # Runnable end-to-end CLI
    ├── classify.py                     # Trivial, Simple, and Few-Shot LLM classifiers
    ├── retrieve.py                     # TF-IDF grounded vector search
    ├── draft.py                        # Grounded reply generator
    ├── route.py                        # Safety guardrails & escalation logic
    ├── taxonomy.py                     # 9-class intent taxonomy & few-shot prompts
    ├── bootstrap_labels.py             # Weak supervision labeling pass
    └── llm.py                          # Multi-provider LLM interface
```

---

## Golden Evaluation Set (200 Hand-Labelled Examples)

As required by Deliverable 2, `eval/golden_set.jsonl` contains **200 hand-labelled examples** built to rigorously evaluate intent classification and safety escalation:

- **180 Stratified Random Samples**: Deterministically drawn across all 9 intent classes from the clean AmazonHelp historical dataset (`data/processed/exchanges.jsonl`), ensuring representative coverage of everyday customer support volume.
- **20 Targeted Edge Cases**: Deliberately sampled and hand-crafted difficult examples including mid-thread contextless fragments (*"I already fill the form"*), sarcastic compliments (*"Shoutout to the driver who threw my package in the rain"*), multi-intent complaints, and high-risk security threats (credential theft, unauthorized card charges).

### Sampling & Labeling Methodology
1. **Independent Evaluation**: Every example was reviewed and labelled for `gold_intent` (from the 9-class taxonomy) and `gold_escalate` (deterministic boolean for human review need).
2. **Quality Filtering**: Non-English tweets, spam bots, and uninterpretable noise were filtered out, maintaining an exact 200-sample benchmark set.
3. **Documentation**: Full guidelines, criteria, and ambiguity rules are documented in [`eval/ANNOTATION_GUIDE.md`](eval/ANNOTATION_GUIDE.md).

