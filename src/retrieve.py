"""
Retrieval over historically-resolved exchanges, used to ground reply
drafting in how the brand actually responded to similar issues before.

Uses TF-IDF + cosine similarity rather than a learned embedding model -
simple, dependency-light, and fast enough for a CLI demo. A real embedding
model (e.g. a sentence-transformer or a hosted embeddings API) is the
obvious upgrade; see report/decision_log.md and the "next week" section of
the report for why this was deferred.
"""

import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class ExchangeRetriever:
    def __init__(self, exchanges: list[dict]):
        self.exchanges = exchanges
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", min_df=1)
        self.matrix = self.vectorizer.fit_transform(e["customer_text"] for e in exchanges)

    @classmethod
    def from_jsonl(cls, path: Path | None = None, exclude_ids: set[str] | None = None) -> "ExchangeRetriever":
        path = path or (PROCESSED_DIR / "exchanges.jsonl")
        with open(path) as f:
            exchanges = [json.loads(l) for l in f]
        if exclude_ids:
            exchanges = [e for e in exchanges if e["exchange_id"] not in exclude_ids]
        return cls(exchanges)

    def retrieve(self, query_text: str, k: int = 3) -> list[dict]:
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [
            {**self.exchanges[i], "similarity": float(sims[i])}
            for i in top_idx
            if sims[i] > 0
        ]