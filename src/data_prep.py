"""
Loads the Customer Support on Twitter dataset and normalizes it into
per-brand conversation threads.

Two sources are supported, tried in this order:

1. Kaggle raw CSV (data/raw/twcs.csv) - the original
   thoughtvector/customer-support-on-twitter schema, if the user has
   downloaded it via the Kaggle API/website. This is the canonical source.
2. Hugging Face mirror (TNE-AI/customer-support-on-twitter-conversation) -
   the same underlying tweets, pre-threaded into Customer:/Support: turns
   and already tagged with a company (brand) column. Used as a fallback
   because the raw Kaggle CSV requires an authenticated Kaggle account and
   cannot be fetched anonymously. See report/decision_log.md.

Output:
  data/processed/threads.jsonl - one JSON object per raw (filtered) thread:
    {"thread_id": str, "brand": str, "turns": [{"speaker": ..., "text": ...}]}

  data/processed/exchanges.jsonl - the atomic unit used by the rest of the
  pipeline. One JSON object per (customer message, brand's reply to it):
    {"exchange_id": str, "brand": str, "customer_text": str, "support_text": str}

We flatten threads down to adjacent customer->support pairs instead of
using full multi-turn threads as context. The HF mirror's conversation_id
groups everyone who replied to a viral root tweet into one "conversation",
so even short, capped threads can interleave different customers (verified
by manual inspection - see report/decision_log.md). A single message and
the reply immediately after it is the most reliable signal available
without real tweet_id chains, and it matches the assignment's framing:
classify one incoming message, ground its reply in similar past
resolutions, decide on escalation for that message.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # deterministic language detection

# Threads longer than this in the HF mirror are almost always reply-storm
# artifacts (many unrelated customers replying to one viral root tweet, all
# grouped under one conversation_id) rather than a real 1:1 support
# back-and-forth. Verified by manual inspection - see report/decision_log.md.
MAX_TURNS = 10

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
KAGGLE_CSV = RAW_DIR / "twcs.csv"

HF_PARQUET_URL = (
    "https://huggingface.co/datasets/TNE-AI/customer-support-on-twitter-conversation"
    "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
HF_CACHE = RAW_DIR / "hf_conversations.parquet"

URL_RE = re.compile(r"https?://\S+")
HANDLE_RE = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    text = URL_RE.sub("", text)
    text = HANDLE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_from_kaggle_csv(brand: str, limit: int) -> list[dict]:
    """Reconstruct threads from the raw twcs.csv via response_tweet_id chains."""
    df = pd.read_csv(KAGGLE_CSV, dtype=str)
    df["text"] = df["text"].fillna("")
    by_id = df.set_index("tweet_id", drop=False)

    brand_replies = df[(df["inbound"] == "False") & (df["author_id"] == brand)]
    threads = []
    for _, reply in brand_replies.iterrows():
        if len(threads) >= limit:
            break
        parent_id = reply.get("in_response_to_tweet_id")
        if pd.isna(parent_id) or parent_id not in by_id.index:
            continue
        turns, seen = [], set()
        cur = by_id.loc[parent_id]
        chain = [cur]
        while pd.notna(cur.get("in_response_to_tweet_id")) and cur["in_response_to_tweet_id"] in by_id.index:
            cur = by_id.loc[cur["in_response_to_tweet_id"]]
            chain.append(cur)
        chain.reverse()
        chain.append(reply)
        for row in chain:
            tid = row["tweet_id"]
            if tid in seen:
                continue
            seen.add(tid)
            speaker = "support" if row["inbound"] == "False" else "customer"
            turns.append({"speaker": speaker, "text": clean_text(row["text"])})
        threads.append({"thread_id": reply["tweet_id"], "brand": brand, "turns": turns})
    return threads


def _download_hf_mirror() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if HF_CACHE.exists():
        return
    print(f"Downloading HF mirror parquet to {HF_CACHE} ...", file=sys.stderr)
    # requests/urllib3 fail cert verification against this sandbox's CA; curl
    # uses the system trust store and works, so shell out to it instead.
    subprocess.run(
        ["curl", "-sL", "--fail", "-o", str(HF_CACHE), HF_PARQUET_URL],
        check=True,
    )


def _is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def _load_from_hf_mirror(brand: str, limit: int) -> list[dict]:
    _download_hf_mirror()
    df = pd.read_parquet(HF_CACHE, columns=["conversation_id", "company", "conversation"])
    df = df[df["company"].str.lower() == brand.lower()]

    turn_re = re.compile(r"(Customer|Support):\s*(.*?)(?=(?:Customer:|Support:)|$)", re.S)
    threads = []
    for _, row in df.iterrows():
        if len(threads) >= limit:
            break
        turns = []
        for speaker_raw, text in turn_re.findall(row["conversation"]):
            cleaned = clean_text(text)
            if not cleaned:
                continue
            turns.append({
                "speaker": "customer" if speaker_raw == "Customer" else "support",
                "text": cleaned,
            })
        if not turns or len(turns) > MAX_TURNS:
            continue  # drop empty and reply-storm-contaminated "threads"
        if not any(t["speaker"] == "support" for t in turns):
            continue  # no resolution to ground on
        first_customer_turn = next((t["text"] for t in turns if t["speaker"] == "customer"), "")
        if not _is_english(first_customer_turn):
            continue
        threads.append({"thread_id": row["conversation_id"], "brand": brand, "turns": turns})
    return threads


def list_hf_brand_counts(top_n: int = 20) -> pd.Series:
    """Utility used once to pick the brand - value counts of company."""
    _download_hf_mirror()
    df = pd.read_parquet(HF_CACHE, columns=["company"])
    return df["company"].value_counts().head(top_n)


def extract_exchanges(threads: list[dict]) -> list[dict]:
    """Flatten each thread into adjacent (customer_text, support_text) pairs."""
    seen_customer_texts = set()
    exchanges = []
    for t in threads:
        turns = t["turns"]
        for i in range(len(turns) - 1):
            if turns[i]["speaker"] == "customer" and turns[i + 1]["speaker"] == "support":
                customer_text = turns[i]["text"]
                if customer_text in seen_customer_texts:
                    continue
                seen_customer_texts.add(customer_text)
                exchanges.append({
                    "exchange_id": f"{t['thread_id']}_{i}",
                    "brand": t["brand"],
                    "customer_text": customer_text,
                    "support_text": turns[i + 1]["text"],
                })
    return exchanges


def build_threads(brand: str, limit: int = 5000) -> tuple[Path, Path]:
    if KAGGLE_CSV.exists():
        threads = _load_from_kaggle_csv(brand, limit)
        source = "kaggle_csv"
    else:
        threads = _load_from_hf_mirror(brand, limit)
        source = "hf_mirror"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    threads_path = PROCESSED_DIR / "threads.jsonl"
    with threads_path.open("w") as f:
        for t in threads:
            f.write(json.dumps(t) + "\n")

    exchanges = extract_exchanges(threads)
    exchanges_path = PROCESSED_DIR / "exchanges.jsonl"
    with exchanges_path.open("w") as f:
        for e in exchanges:
            f.write(json.dumps(e) + "\n")

    print(
        f"brand={brand} source={source} -> {len(threads)} threads, "
        f"{len(exchanges)} exchanges ({threads_path}, {exchanges_path})"
    )
    return threads_path, exchanges_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", default="AmazonHelp")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--list-brands", action="store_true")
    args = parser.parse_args()

    if args.list_brands:
        print(list_hf_brand_counts(30))
    else:
        build_threads(args.brand ,args.limit)







