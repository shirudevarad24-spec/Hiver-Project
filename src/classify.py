"""
Three intent classifiers, in increasing sophistication - used both in the
real pipeline (llm_classify) and as baselines the report compares against.
  trivial_classify - always predicts the single most common training label.
  simple_classify  - TF-IDF + logistic regression, trained on the
                     LLM-bootstrapped pseudo-labels (see bootstrap_labels.py).
  llm_classify     - few-shot prompt against the taxonomy, one message at a
                     time, returns intent + confidence + rationale.
"""

import json
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from llm import complete_json
from taxonomy import INTENT_NAMES, INTENTS

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

TAXONOMY_BLOCK = "\n".join(f"- {name}: {info['description']}" for name, info in INTENTS.items())
FEW_SHOT_BLOCK = "\n".join(
    f'  ("{ex}", "{name}")' for name, info in INTENTS.items() for ex in info["examples"][:1]
)

SYSTEM = f"""You classify a single customer support tweet sent to Amazon into \
exactly one of these intents:
{TAXONOMY_BLOCK}

Examples:
{FEW_SHOT_BLOCK}

Respond as JSON: {{"intent": <name>, "confidence": <0-1 float>, "rationale": <one short sentence>}}."""


class TrivialClassifier:
    """Baseline #1: always guess the majority class from training labels."""

    def __init__(self, train_labels: list[str]):
        self.majority = Counter(train_labels).most_common(1)[0][0]

    def predict(self, text: str) -> str:
        return self.majority


class SimpleClassifier:
    """Baseline #2: TF-IDF + logistic regression on bootstrap pseudo-labels."""

    def __init__(self, texts: list[str], labels: list[str]):
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", min_df=2)
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model.fit(X, labels)

    def predict(self, text: str) -> str:
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

    @classmethod
    def from_bootstrap_file(cls, path: Path | None = None) -> "SimpleClassifier":
        path = path or (PROCESSED_DIR / "bootstrap_labels.jsonl")
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        return cls([r["customer_text"] for r in rows], [r["intent"] for r in rows])


def llm_classify(text: str) -> dict:
    result = complete_json(SYSTEM, text, max_tokens=200)
    if result.get("intent") not in INTENT_NAMES:
        result["intent"] = "service_complaint_generic"  # safe fallback bucket
    return result
