"""
main.py

Functionality:
1. Load raw tweet dataset (data/raw/stock_tweets.csv)
2. Clean tweets → save data/processed/cleaned_tweets.csv
3. Compute TextBlob sentiment
4. Aggregate weekly average sentiment per ticker → save data/processed/weekly_sentiment.csv
5. Create weekly heatmap (Ticker × Week) → save data/processed/weekly_sentiment_heatmap.png
6. Build / load tweet embeddings (for semantic search)
7. Start an Ollama-powered chat to answer questions about the tweets

Run with:
    python main.py
"""

import os
import re
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import ollama  # pip install ollama


# ------------------------------------------------------------
# Paths & config
# ------------------------------------------------------------
RAW_CSV_PATH = "data/raw/stock_tweets.csv"
PROCESSED_DIR = "data/processed"

CLEANED_CSV = os.path.join(PROCESSED_DIR, "cleaned_tweets.csv")
WEEKLY_CSV = os.path.join(PROCESSED_DIR, "weekly_sentiment.csv")
HEATMAP_PNG = os.path.join(PROCESSED_DIR, "weekly_sentiment_heatmap.png")
EMBED_CACHE_PATH = os.path.join(PROCESSED_DIR, "tweet_embeddings.npz")

CHAT_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"
TOP_K = 25


# ------------------------------------------------------------
# Helper Funtions: cleaning
# ------------------------------------------------------------
def clean_tweet(text: str) -> str:
    """
    Cleans Tweets for things like URLs, @mentions, non-alphanumeric, etc
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------------------------------------------------
# Helper Funtions: sentiment
# ------------------------------------------------------------
def get_polarity(text: str) -> float:
    """
    TextBlob: 1 = positive sentiment, -1 = negative sentiment
    """
    return TextBlob(text).sentiment.polarity


# ------------------------------------------------------------
# Helper Funtions: embeddings & semantic search
# ------------------------------------------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts using Ollama's embeddings API.
    """
    embeddings = []
    for i, t in enumerate(texts):
        t = t.strip()
        if not t:
            embeddings.append(None)
            continue

        resp = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        vec = np.array(resp["embedding"], dtype=np.float32)
        embeddings.append(vec)

        if (i + 1) % 500 == 0:
            print(f"  Embedded {i+1} tweets...")

    dim = len(embeddings[0])
    for idx, v in enumerate(embeddings):
        if v is None:
            embeddings[idx] = np.zeros(dim, dtype=np.float32)

    return np.vstack(embeddings)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single vector a and rows of b.
    """
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


def ensure_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Ensure there is an embeddings for all tweets.

    If EMBED_CACHE_PATH exists and matches number of rows, load it.
    else, compute and cache (note computing can take a while).
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if os.path.exists(EMBED_CACHE_PATH):
        print(f"Loading cached embeddings from {EMBED_CACHE_PATH} ...")
        data = np.load(EMBED_CACHE_PATH)
        embeddings = data["embeddings"]
        if embeddings.shape[0] == len(df):
            return embeddings
        else:
            print("Cached embeddings do not match tweet count. Recomputing...")

    print("Computing embeddings with Ollama (this may take a while)...")
    texts = df["clean_text"].tolist()
    embeddings = embed_texts(texts)
    np.savez_compressed(EMBED_CACHE_PATH, embeddings=embeddings)
    print(f"Saved embeddings → {EMBED_CACHE_PATH}")
    return embeddings


def search_tweets(query: str, df: pd.DataFrame, embeddings: np.ndarray, top_k: int = TOP_K) -> pd.DataFrame:
    """
    Given a user query, find top_k most semantically relevant tweets.
    """
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=query)
    qvec = np.array(resp["embedding"], dtype=np.float32)

    sims = cosine_sim(qvec, embeddings)
    top_idx = np.argsort(-sims)[:top_k]
    return df.iloc[top_idx].copy()


# ------------------------------------------------------------
# Helper Funtions: chatting with Ollama
# ------------------------------------------------------------
def build_context_snippet(rows: pd.DataFrame) -> str:
    """
    Build a context string summarizing the most relevant tweets.
    """
    lines = []
    for _, r in rows.iterrows():
        dt = r["tweet_datetime"]
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M")
        ticker = r["ticker"]
        pol = r["polarity"]
        text = r["text"]
        if len(text) > 240:
            text = text[:240] + "..."
        lines.append(
            f"- [{date_str} {time_str}] Ticker {ticker}, polarity={pol:.3f}: {text}"
        )
    return "\n".join(lines)


def ask_ollama(question: str, context: str) -> str:
    """
    Send question + tweet context to Ollama chat model and return the answer.
    """
    system_prompt = (
        "You are an assistant that answers questions about stock-related tweets. "
        "Use ONLY the tweet context provided below to answer. "
        "If the answer is unclear from the tweets, say that the tweets do not clearly explain it.\n\n"
        "Context tweets:\n"
        f"{context}\n\n"
        "Base your reasoning entirely on these tweets."
    )

    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return resp["message"]["content"]


# ------------------------------------------------------------
# Load, clean, process, aggregate, heatmap
# ------------------------------------------------------------
def run_pipeline() -> pd.DataFrame:
    """
    Do preprocessing:
    - Load raw data
    - Clean & compute sentiment
    - Save cleaned and weekly sentiment CSVs
    - Generate & save heatmap

    Returns:
        df: full tweet dataframe (with clean_text, polarity, tweet_datetime, ticker)
    """
    if not os.path.exists(RAW_CSV_PATH):
        print(f"ERROR: {RAW_CSV_PATH} not found.")
        sys.exit(1)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load raw
    print(f"Loading {RAW_CSV_PATH} ...")
    df = pd.read_csv(RAW_CSV_PATH)

    df = df.rename(columns={
        "Tweet": "text",
        "Stock Name": "ticker",
        "Date": "tweet_datetime"
    })
    df["tweet_datetime"] = pd.to_datetime(df["tweet_datetime"])

    # Clean
    print("Cleaning tweets...")
    df["clean_text"] = df["text"].apply(clean_tweet)
    df = df[df["clean_text"].str.len() > 0].copy()

    # Sentiment
    print("Computing sentiment polarity...")
    df["polarity"] = df["clean_text"].apply(get_polarity)

    # Save cleaned data
    df.to_csv(CLEANED_CSV, index=False)
    print(f"Saved cleaned tweets → {CLEANED_CSV}")

    # Weekly aggregation
    df["tweet_week"] = df["tweet_datetime"].dt.to_period("W").dt.start_time
    weekly_sentiment = (
        df.groupby(["ticker", "tweet_week"])["polarity"]
          .mean()
          .reset_index()
          .rename(columns={"polarity": "avg_weekly_polarity"})
    )

    weekly_sentiment.to_csv(WEEKLY_CSV, index=False)
    print(f"Saved weekly sentiment → {WEEKLY_CSV}")

    print("\nWeekly sentiment preview:")
    print(weekly_sentiment.head())

    # Heatmap
    print("\nGenerating heatmap...")
    matrix = weekly_sentiment.pivot(
        index="ticker",
        columns="tweet_week",
        values="avg_weekly_polarity"
    ).fillna(0.0)

    print("Matrix shape:", matrix.shape)

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest"
    )

    # Y-axis tickers
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    # X-axis weeks (downsample labels)
    weeks = matrix.columns
    n_weeks = len(weeks)
    max_labels = 30
    step = max(1, n_weeks // max_labels)
    x_positions = np.arange(0, n_weeks, step)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [weeks[i].strftime("%Y-%m-%d") for i in x_positions],
        rotation=45,
        ha="right"
    )

    ax.set_xlabel("Week")
    ax.set_ylabel("Ticker")
    ax.set_title("Ticker × Week Heatmap of Avg Weekly Tweet Sentiment")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Avg Weekly Polarity (-1 negative → +1 positive)")

    plt.tight_layout()
    plt.savefig(HEATMAP_PNG, dpi=300)
    print(f"Saved heatmap → {HEATMAP_PNG}")
    plt.close(fig)

    return df


# ------------------------------------------------------------
# 4. Chat loop
# ------------------------------------------------------------
def run_chat(df: pd.DataFrame) -> None:
    """
    Start an interactive chat over the tweet dataset using Ollama.
    """
    print("\nBuilding / loading embeddings for chat...")
    embeddings = ensure_embeddings(df)

    print(
        "\n=== Tweet QA Chat (Ollama) ===\n"
        "Ask questions about the tweets, for example:\n"
        "  Why did ticker F have negative sentiment on May 23, 2022?\n"
        "  What were people saying about TSLA in early October 2021?\n"
        "Type 'exit' or 'quit' to leave.\n"
    )

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        print("Searching relevant tweets...")
        hits = search_tweets(question, df, embeddings, top_k=TOP_K)
        if hits.empty:
            print("No relevant tweets found for that question.")
            continue

        context = build_context_snippet(hits)

        print("Asking Ollama...")
        answer = ask_ollama(question, context)
        print("\nAssistant:", answer, "\n")


# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------
def main():
    df = run_pipeline()
    run_chat(df)


if __name__ == "__main__":
    main()
