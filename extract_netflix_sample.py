"""
Netflix Prize Dataset - Sample Extractor
=========================================
Run this script from the folder where your Netflix Prize files are located.

It will produce:
  - ratings_sample.csv     (~2M rows, all 4 combined_data files sampled evenly)
  - movie_titles_clean.csv (all 17,770 movies, cleaned)

Usage:
    python extract_netflix_sample.py

Requirements:
    pip install pandas
"""

import os
import pandas as pd
import random

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILES = [
    "combined_data_1.txt",
    "combined_data_2.txt",
    "combined_data_3.txt",
    "combined_data_4.txt",
]
MOVIE_TITLES_FILE = "movie_titles.csv"

TARGET_ROWS      = 2_000_000   # total ratings rows to sample
SAMPLE_SEED      = 42          # for reproducibility
OUTPUT_RATINGS   = "ratings_sample.csv"
OUTPUT_MOVIES    = "movie_titles_clean.csv"
# ─────────────────────────────────────────────────────────────────────────────


def parse_and_sample(filepath, target_per_file):
    """
    Parse one combined_data file and reservoir-sample target_per_file rows.
    The file format is:
        MovieID:
        UserID,Rating,Date
        UserID,Rating,Date
        ...
    Returns a list of [movie_id, user_id, rating, date] lists.
    """
    print(f"  Reading {filepath} ...")
    reservoir = []
    count     = 0
    movie_id  = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Movie header line
            if line.endswith(":"):
                movie_id = int(line[:-1])
                continue

            # Rating line
            parts = line.split(",")
            if len(parts) != 3:
                continue

            user_id, rating, date = parts
            row = [movie_id, int(user_id), int(rating), date]
            count += 1

            # Reservoir sampling
            if len(reservoir) < target_per_file:
                reservoir.append(row)
            else:
                j = random.randint(0, count - 1)
                if j < target_per_file:
                    reservoir[j] = row

    print(f"    → {count:,} total rows found, kept {len(reservoir):,}")
    return reservoir


def load_movie_titles(filepath):
    """
    Load movie_titles.csv. The file has no header and uses:
        MovieID,Year,Title
    Some titles contain commas, so we limit to 2 splits max.
    """
    print(f"\nReading {filepath} ...")
    rows = []
    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            parts = line.split(",", 2)   # max 2 splits → 3 parts
            if len(parts) == 3:
                movie_id, year, title = parts
                rows.append({
                    "movie_id": int(movie_id),
                    "year":     year.strip(),   # may be "NULL"
                    "title":    title.strip(),
                })
    df = pd.DataFrame(rows)
    print(f"  → {len(df):,} movies loaded")
    return df


def main():
    random.seed(SAMPLE_SEED)

    # ── Check files exist ────────────────────────────────────────────────────
    missing = [f for f in DATA_FILES + [MOVIE_TITLES_FILE] if not os.path.exists(f)]
    if missing:
        print("\n⚠  The following files were not found in the current directory:")
        for m in missing:
            print(f"   - {m}")
        print("\nPlease run this script from the folder containing the Netflix Prize files.")
        return

    # ── Sample ratings ───────────────────────────────────────────────────────
    per_file = TARGET_ROWS // len(DATA_FILES)
    print(f"\nSampling {per_file:,} rows from each of {len(DATA_FILES)} files "
          f"(target total: {TARGET_ROWS:,})\n")

    all_rows = []
    for filepath in DATA_FILES:
        all_rows.extend(parse_and_sample(filepath, per_file))

    ratings_df = pd.DataFrame(all_rows, columns=["movie_id", "user_id", "rating", "date"])
    ratings_df["date"] = pd.to_datetime(ratings_df["date"])
    ratings_df = ratings_df.sample(frac=1, random_state=SAMPLE_SEED).reset_index(drop=True)

    ratings_df.to_csv(OUTPUT_RATINGS, index=False)
    print(f"\n✅  Ratings saved → {OUTPUT_RATINGS}  ({len(ratings_df):,} rows)")
    print(f"    Columns : {list(ratings_df.columns)}")
    print(f"    Size    : {os.path.getsize(OUTPUT_RATINGS) / 1e6:.1f} MB")

    # ── Movie titles ─────────────────────────────────────────────────────────
    movies_df = load_movie_titles(MOVIE_TITLES_FILE)
    movies_df.to_csv(OUTPUT_MOVIES, index=False)
    print(f"\n✅  Movies saved  → {OUTPUT_MOVIES}  ({len(movies_df):,} rows)")
    print(f"    Columns : {list(movies_df.columns)}")
    print(f"    Size    : {os.path.getsize(OUTPUT_MOVIES) / 1e6:.1f} MB")

    # ── Quick sanity check ───────────────────────────────────────────────────
    print("\n── Ratings sample preview ──────────────────────────────────────")
    print(ratings_df.head(5).to_string(index=False))
    print(f"\nUnique users  : {ratings_df['user_id'].nunique():,}")
    print(f"Unique movies : {ratings_df['movie_id'].nunique():,}")
    print(f"Rating range  : {ratings_df['rating'].min()} – {ratings_df['rating'].max()}")
    print(f"Date range    : {ratings_df['date'].min().date()} → {ratings_df['date'].max().date()}")

    print("\n── Movies preview ──────────────────────────────────────────────")
    print(movies_df.head(5).to_string(index=False))
    print("\nDone! Upload both CSV files to Claude.")


if __name__ == "__main__":
    main()
