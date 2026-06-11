from pathlib import Path
import requests
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"

BASICS_PATH = DATA_DIR / "title.basics.tsv.gz"
RATINGS_PATH = DATA_DIR / "title.ratings.tsv.gz"
OUTPUT_PATH = DATA_DIR / "imdb_top250.csv"

# Raised to 200k to match real IMDB Top 250 quality
MIN_VOTES = 200_000


# ============================================================
# STEP 1 — Download IMDb datasets
# ============================================================
def download_file(url: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Already downloaded: {output_path}")
        return

    print(f"Downloading: {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end="")

    print(f"\nSaved: {output_path}")


# ============================================================
# STEP 2 — Load and clean datasets
# ============================================================
def load_movie_basics(path: Path) -> pd.DataFrame:
    print("Loading title.basics.tsv.gz — this takes 30-60 seconds, don't panic...")

    basics = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        na_values="\\N",
        usecols=[
            "tconst",
            "titleType",
            "primaryTitle",
            "isAdult",
            "startYear",
            "runtimeMinutes",
            "genres",
        ],
        dtype={
            "tconst": "string",
            "titleType": "string",
            "primaryTitle": "string",
            "isAdult": "string",
            "startYear": "string",
            "runtimeMinutes": "string",
            "genres": "string",
        },
    )

    print(f"Raw rows loaded: {len(basics):,}")

    # Keep only non-adult movies
    basics = basics[
        (basics["titleType"] == "movie") &
        (basics["isAdult"] == "0")
    ].copy()

    print(f"After filtering movies only: {len(basics):,}")

    # Convert year and runtime to numbers
    basics["year"] = pd.to_numeric(basics["startYear"], errors="coerce")
    basics["runtime_minutes"] = pd.to_numeric(basics["runtimeMinutes"], errors="coerce")

    # Remove rows with missing title or year
    basics = basics.dropna(subset=["primaryTitle", "year"])

    print(f"After dropping missing values: {len(basics):,}")
    return basics


def load_ratings(path: Path) -> pd.DataFrame:
    print("Loading title.ratings.tsv.gz...")

    ratings = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        dtype={
            "tconst": "string",
            "averageRating": "float32",
            "numVotes": "int32",
        },
    )

    print(f"Ratings loaded: {len(ratings):,} titles")
    return ratings


# ============================================================
# STEP 3 — Create Top 250 ranking
# ============================================================
def create_top250(basics: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted rating formula (same logic IMDB uses):
    weighted_score = (v / (v + m)) * R + (m / (v + m)) * C

    R = movie average rating
    v = number of votes for the movie
    m = minimum votes threshold
    C = mean rating across all eligible movies
    """
    print("\nMerging basics and ratings...")
    movies = basics.merge(ratings, on="tconst", how="inner")
    print(f"After merge: {len(movies):,} movies")

    # Filter by minimum votes
    movies = movies[movies["numVotes"] >= MIN_VOTES].copy()
    print(f"After MIN_VOTES ({MIN_VOTES:,}) filter: {len(movies):,} movies")

    if movies.empty:
        raise ValueError("No movies found. Try lowering MIN_VOTES.")

    # Weighted score calculation
    C = movies["averageRating"].mean()
    m = MIN_VOTES

    movies["weighted_score"] = (
        (movies["numVotes"] / (movies["numVotes"] + m)) * movies["averageRating"]
        + (m / (movies["numVotes"] + m)) * C
    )

    # Sort and take top 250
    movies = movies.sort_values(
        by=["weighted_score", "numVotes"],
        ascending=[False, False]
    ).head(250).reset_index(drop=True)

    movies.insert(0, "rank", movies.index + 1)

    # Build clean output dataframe
    final_df = pd.DataFrame({
        "rank":             movies["rank"],
        "title":            movies["primaryTitle"],
        "year":             movies["year"].astype("int64"),
        "rating":           movies["averageRating"].round(1),
        "votes":            movies["numVotes"],
        "genre":            movies["genres"].fillna("N/A").str.replace(",", ", ", regex=False),
        "runtime_minutes":  movies["runtime_minutes"],
        "imdb_id":          movies["tconst"],
        "imdb_url":         "https://www.imdb.com/title/" + movies["tconst"].astype(str) + "/",
        "weighted_score":   movies["weighted_score"].round(4),
    })

    return final_df


# ============================================================
# STEP 4 — Main runner
# ============================================================
def main() -> None:
    print("=" * 50)
    print("CriticLens — IMDb Top 250 Dataset Builder")
    print("=" * 50 + "\n")

    download_file(BASICS_URL, BASICS_PATH)
    download_file(RATINGS_URL, RATINGS_PATH)

    basics  = load_movie_basics(BASICS_PATH)
    ratings = load_ratings(RATINGS_PATH)

    top250  = create_top250(basics, ratings)

    top250.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDone. Saved {len(top250)} movies to: {OUTPUT_PATH}")
    print("\nTop 10 Preview:")
    print(top250[["rank", "title", "year", "rating", "votes", "genre"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()