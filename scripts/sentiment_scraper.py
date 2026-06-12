import os
import time
import random
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============================================================
# SETUP
# ============================================================
load_dotenv()
API_KEY = os.getenv("OMDB_API_KEY")

if not API_KEY:
    raise ValueError("OMDB_API_KEY not found. Check your .env file.")

INPUT_PATH  = Path("data/raw/imdb_top250.csv")
OUTPUT_PATH = Path("data/processed/imdb_with_sentiment.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

analyzer = SentimentIntensityAnalyzer()


# ============================================================
# STEP 1 — Fetch OMDb Data Per Movie
# ============================================================
def fetch_omdb(imdb_id: str, title: str) -> dict:
    """
    Fetches movie metadata and plot from OMDb API using IMDB ID.
    Plot is used as our critic text input for sentiment analysis.
    """
    url = "http://www.omdbapi.com/"
    params = {
        "i": imdb_id,
        "apikey": API_KEY,
        "plot": "full"  # Get full plot description
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get("Response") == "True":
            return {
                "plot":          data.get("Plot", "N/A"),
                "awards":        data.get("Awards", "N/A"),
                "metascore":     data.get("Metascore", "N/A"),
                "imdb_rating":   data.get("imdbRating", "N/A"),
                "box_office":    data.get("BoxOffice", "N/A"),
                "director":      data.get("Director", "N/A"),
                "language":      data.get("Language", "N/A"),
                "country":       data.get("Country", "N/A"),
            }
        else:
            print(f"  OMDb miss for: {title} — {data.get('Error', 'Unknown error')}")
            return {}

    except Exception as e:
        print(f"  Error fetching {title}: {e}")
        return {}


# ============================================================
# STEP 2 — Run VADER Sentiment
# ============================================================
def analyze_sentiment(text: str) -> dict:
    """
    Runs VADER sentiment analysis on input text.
    Returns compound score, positive, negative, neutral scores.

    Compound score ranges from -1 (most negative) to +1 (most positive).
    Intensity = absolute value of compound (how strong the sentiment is).
    """
    if not text or text == "N/A":
        return {
            "sentiment_compound": None,
            "sentiment_positive": None,
            "sentiment_negative": None,
            "sentiment_neutral":  None,
            "sentiment_intensity": None,
            "sentiment_label":    "unknown"
        }

    scores = analyzer.polarity_scores(text)

    # Sentiment label
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "sentiment_compound":  round(compound, 4),
        "sentiment_positive":  round(scores["pos"], 4),
        "sentiment_negative":  round(scores["neg"], 4),
        "sentiment_neutral":   round(scores["neu"], 4),
        "sentiment_intensity": round(abs(compound), 4),
        "sentiment_label":     label
    }


# ============================================================
# STEP 3 — Main Runner
# ============================================================
def main():
    print("=" * 50)
    print("CriticLens — Sentiment Scraper")
    print("=" * 50 + "\n")

    # Load IMDB Top 250
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} movies from {INPUT_PATH}\n")

    results = []

    for i, row in df.iterrows():
        title    = row["title"]
        imdb_id  = row["imdb_id"]

        print(f"[{i+1}/250] Processing: {title}")

        # Fetch OMDb data
        omdb_data = fetch_omdb(imdb_id, title)

        # Run sentiment on plot text
        plot_text = omdb_data.get("plot", "N/A")
        sentiment = analyze_sentiment(plot_text)

        # Merge everything
        combined = {
            **row.to_dict(),
            **omdb_data,
            **sentiment
        }

        results.append(combined)

        # Checkpoint save every 50 movies
        if (i + 1) % 50 == 0:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_path = Path(f"data/processed/checkpoint_{i+1}.csv")
            checkpoint_df.to_csv(checkpoint_path, index=False)
            print(f"\nCheckpoint saved at {i+1} movies\n")

        # Delay to respect API rate limits
        time.sleep(random.uniform(0.3, 0.8))

    # Save final output
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDone. Saved {len(final_df)} movies to {OUTPUT_PATH}")
    print(f"\nSentiment distribution:")
    print(final_df["sentiment_label"].value_counts())
    print(f"\nPreview:")
    print(final_df[["title", "year", "rating", "sentiment_compound", "sentiment_label"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()