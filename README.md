# Recommendation Systems for Personalized Content Discovery

A comparative study of three recommendation approaches on the **Netflix Prize** dataset:
a **bias baseline**, **matrix factorization (biased SVD)**, and **item-based collaborative
filtering**. Models are evaluated on both rating accuracy (**RMSE**) and ranking quality
(**MAP@10**, relevance = rating ≥ 3.5), along with MAE, Precision@10, Recall@10, NDCG@10,
and catalog coverage.

## Key Result

| Model | RMSE | MAP@10 (sampled) | HR@10 | Coverage |
|---|---|---|---|---|
| Baseline / Popularity | **0.915** | **0.107** | **0.502** | 0.2% |
| Matrix Factorization | **0.915** | 0.088 | 0.452 | 0.4% |
| Item-based CF | 1.012 | 0.019 | 0.126 | **61.0%** |

**Takeaway:** RMSE and ranking quality measure different things. On sparse, temporally-split
data the bias terms capture most of the predictable *rating* signal, so RMSE barely separates
models — yet they behave very differently when *ranking* a catalog. Popularity is a deceptively
strong hit-rate baseline; matrix factorization is the best-balanced personalized choice;
item-based CF trades accuracy for diversity. **No single model wins every metric.**

## Repository Structure

```
.
├── README.md                  # this file
├── requirements.txt           # Python dependencies
├── extract_netflix_sample.py  # build a manageable CSV sample from the raw Netflix files
├── netflix_recommender.ipynb  # full annotated analysis notebook (EDA → models → eval → recs)
├── run_pipeline.py            # CLI: run the whole pipeline from the terminal
├── src/
│   ├── data.py                # loading, k-core filtering, temporal train/test split
│   ├── models.py              # BiasModel, MatrixFactorization, ItemCF
│   ├── evaluate.py            # RMSE/MAE + MAP@10/P@10/R@10/NDCG@10/coverage
│   └── recommend.py           # Top-K generation + item-based explanations
├── data/                      # place the CSV files here (git-ignored)
└── reports/
    ├── Technical_Report.pdf   # the written report
    └── Presentation.pdf       # the slide deck
```

## Setup

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt
```

## Getting the Data

The raw Netflix Prize files are large (~2 GB total). This project works on a manageable
**2-million-rating sample**.

1. Download the dataset from
   [Kaggle: Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).
2. Run the extractor in the folder containing the raw `combined_data_*.txt` and
   `movie_titles.csv` files:
   ```bash
   python extract_netflix_sample.py
   ```
   This produces `ratings_sample.csv` (~2M ratings) and `movie_titles_clean.csv`.
3. Move both CSVs into this project's `data/` folder.

> The sample uses reservoir sampling with a fixed seed, so it is representative and reproducible.

## Running

**Option A — Jupyter notebook (recommended for exploration):**
```bash
jupyter notebook netflix_recommender.ipynb
```
Run all cells top to bottom. The notebook covers EDA, preprocessing, all three models,
both evaluation protocols, Top-K recommendations, and explainability.

**Option B — command line (full pipeline):**
```bash
python run_pipeline.py --ratings data/ratings_sample.csv --movies data/movie_titles_clean.csv
```

## Method Summary

- **Preprocessing:** iterative *k-core* filtering (users ≥ 15 ratings, movies ≥ 10) to obtain a
  dense, evaluable subset (~630K ratings, 28K users, 6.8K movies).
- **Split:** per-user **temporal** hold-out (most recent 20% of each user's ratings → test),
  mirroring the real "predict the future" task.
- **Models:** regularized bias baseline; biased SVD trained with SGD (15 factors, early-stopped
  at ~10 epochs); item-based CF with cosine similarity over item-baseline-centred vectors,
  positive-only top-50 neighbours.
- **Evaluation:** RMSE/MAE on held-out ratings; MAP@10 (relevance ≥ 3.5) plus Precision/Recall/
  NDCG@10 and coverage, under both an *all-items* and a *100-sampled-negative* ranking protocol.

## Reproducibility

All randomness is seeded (`seed = 42`). The notebook and `run_pipeline.py` produce the numbers
reported in `reports/Technical_Report.pdf`.

## Future Work

Scale to the full 100M ratings; implicit-feedback and ranking-native models (ALS, BPR,
learning-to-rank); hybrid/content features for cold-start; diversity-aware re-ranking; neural CF.
