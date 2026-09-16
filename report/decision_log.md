# Engineering Decision Log

A candid list of 12 non-obvious decisions, rejected alternatives, and operational trade-offs made while building this support agent:

* **1. Choosing 9 operational intents instead of Banking77's 77 classes**  
  *Alternative rejected:* Adopting an off-the-shelf fine-grained taxonomy like Banking77.  
  *Why:* Twitter customer support interactions are noisy, coarse-grained, and fast-paced. Operational workflows only care about the next concrete action (e.g. redirect to self-serve portal vs. request DM vs. escalate to security). 9 well-separated categories map 1:1 to operational routing decisions and avoid fine-grained classification noise.

* **2. Using in-memory TF-IDF sparse retrieval instead of a heavy vector database**  
  *Alternative rejected:* Deploying ChromaDB, Pinecone, or Faiss with heavy dense embeddings.  
  *Why:* For ~8,000 historical AmazonHelp tweets, in-memory TF-IDF cosine similarity executes in sub-millisecond latency with zero external database dependencies. Crucially, Twitter support relies heavily on exact keyword tokens (specific URLs like `amazon.com/help`, rep signoffs like `^CS`, and order IDs) which sparse retrieval preserves without embedding distortion.

* **3. Enforcing a strict anti-leakage filter (`exclude_ids`) during all retrieval evaluations**  
  *Alternative rejected:* Querying the entire historical corpus blindly.  
  *Why:* If an evaluation tweet's historical exchange exists in the corpus, nearest-neighbor retrieval will trivially pull its own paired answer with ~1.0 cosine similarity, artificially inflating groundedness scores to 100%. We explicitly pass `exclude_ids` in all evaluation scripts and demo loops to guarantee true generalization.

* **4. Placing deterministic regex safety guardrails before LLM routing checks**  
  *Alternative rejected:* Letting a single LLM prompt decide both the drafted reply and whether to escalate.  
  *Why:* LLMs are probabilistic systems. Even at temperature 0, an adversarial prompt or unusual phrasing can lead to hallucinated compliance. Legal threats (*"sue"*, *"lawyer"*, *"police"*) and security compromises (*"hacked"*, *"unauthorized charge"*) must have 100% deterministic escalation guarantees.

* **5. Hardcoding an `ALWAYS_ESCALATE` policy for high-risk financial and security intents**  
  *Alternative rejected:* Allowing the model to auto-reply to billing or account security queries if confidence is high.  
  *Why:* Twitter accounts can be spoofed and tweets are public. An automated agent should never attempt to autonomously resolve account takeovers or billing disputes over public social media timelines.

* **6. Implementing an "Ungrounded Hallucination Trap" using a cosine similarity floor**  
  *Alternative rejected:* Letting the LLM draft a reply regardless of retrieval relevance.  
  *Why:* When a customer presents an unprecedented or out-of-distribution issue, top retrieval similarity drops below 0.30. In this case, generation is aborted and the ticket is routed to human triage with the stated reason `"ungrounded draft"`, preventing fabricated policies.

* **7. Building a two-tier baseline system (Trivial + Simple) alongside the LLM**  
  *Alternative rejected:* Evaluating only the final LLM pipeline in isolation.  
  *Why:* A 74% accuracy figure is meaningless without knowing the floor. The majority-class trivial baseline (26.1%) exposes the class imbalance, while the TF-IDF + Logistic Regression baseline (69.6%) proves how much performance can be achieved without spending any LLM tokens.

* **8. Isolating single-turn initial tweets for Tier-1 triage MVP**  
  *Alternative rejected:* Ingesting full 10-turn multi-day Twitter threads into the prompt context.  
  *Why:* Over 85% of Twitter customer service interactions begin with a public tweet where the immediate goal is routing or link deflection within a 5-minute SLA. Building a single-turn agent isolates the highest-impact entry point with a minimal failure surface.

* **9. Double-blind human scoring for LLM-as-a-Judge calibration**  
  *Alternative rejected:* Accepting LLM judge scores as objective ground truth.  
  *Why:* Having an LLM judge evaluate LLM-generated text risks a self-referential feedback loop. We hand-scored a test sample blindly in `eval/human_scores.jsonl` and computed Pearson correlation ($r > 0.85$) across all four rubric dimensions to empirically prove the judge reflects human quality standards.

* **10. Prioritizing recall over precision on escalation routing**  
  *Alternative rejected:* Optimizing for maximum automated deflection rate.  
  *Why:* A False Positive escalation costs an agent 20 seconds to triage and close. A False Negative escalation (sending a bot link to an enraged user whose account was emptied) causes severe churn and public brand damage. The routing engine is intentionally tuned to be conservative.

* **11. Strip-formatting leading @mentions during data cleaning while keeping URLs**  
  *Alternative rejected:* Keeping raw tweet strings or removing all punctuation/links.  
  *Why:* Customer tweets often tag multiple irrelevant handles (`@AmazonHelp @amazon @JeffBezos`). Stripping leading handles prevents the classifier from learning spurious handle correlations, while preserving URLs and order IDs is critical for grounded retrieval.

* **12. Provider-agnostic LLM client with token capping and rate-limit backoff**  
  *Alternative rejected:* Hardcoding the pipeline to a single proprietary SDK or model ID.  
  *Why:* Evaluators test on different machines and API tiers (Groq, Anthropic, xAI). Implementing automatic key detection, `.env` loading, output token capping (800 tokens to avoid Groq OTPM limits), and exponential backoff guarantees that the evaluation suite runs out-of-the-box without crashing.
