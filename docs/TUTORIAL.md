# Tutorial: data layer and custom algorithms

Hands-on guide for the CSV data layer and building custom ranking experiments on V3 features.

Install and run the built-in pipelines first: [Getting started](GETTING_STARTED.md).

There is no SQL database — all inputs and outputs are CSV files under `recsocial_py/data/` and `recsocial_py/reports/`.

---

## Granular CLI

Inspect intermediate files between pipeline stages:

```bash
# V1
recsocial v1 preprocess
recsocial v1 score
recsocial v1 experiment

# V2 (requires V1 preprocess)
recsocial v1 preprocess
recsocial v2 preprocess
recsocial v2 score
recsocial v2 experiment

# V3
recsocial v3 preprocess
recsocial v3 score
recsocial v3 experiment
```

## Python API

```python
from recsocial.shared.pipeline import package_root, run_all_pipelines, run_v3_pipeline

root = package_root()
paths = run_all_pipelines(root)          # all versions
v3_paths = run_v3_pipeline(root)         # V3 only
```

## Output files

| Version | Report | Recommendations | Metrics |
|---------|--------|-----------------|---------|
| V1 | `reports/v1/report.md` | `rerank_recommendations.csv` | `trial_metrics_summary.csv` |
| V2 | `reports/v2/report.md` | `v2_recommendations.csv` | `v2_metrics_summary.csv` |
| V3 | `reports/v3/report.md` | `v3_recommendations.csv` | `v3_metrics_summary.csv` |

Validation: `reports/validation_summary.md` — see [Evaluation](EVALUATION.md).

---

## Data layer

### Pipeline

```text
data/raw/
    ▼ V1 preprocess
data/v1/processed/          users, news, ratings, comments
data/v1/interim/            scored_news_sc.csv, scored_news_scsa.csv
    ▼ V2 preprocess + score
data/v2/processed/          news_enriched.csv, users.csv, ratings.csv
data/v2/interim/            component_scores.csv
    ▼ V3 score
data/v3/processed/          ratings.csv, user_strength.csv
data/v3/interim/            v3_feature_scores.csv
    ▼ experiments
reports/v1|v2|v3/
```

Generated paths are gitignored. Regenerate with `recsocial run all`.

### Key files

| Path | Use |
|------|-----|
| `data/v3/interim/v3_feature_scores.csv` | One row per tweet — all V2 + V3 features |
| `data/v3/processed/ratings.csv` | Normalized trial sessions |
| `data/v2/interim/component_scores.csv` | V2 components only |
| `data/raw/paper_rankings/` | Author ranking exports (V2 reproduction) |

Join key: `news_id` (cast to string on both sides).

### `v3_feature_scores.csv` columns

| Group | Examples |
|-------|----------|
| Identity | `news_id`, `author_id`, `text` |
| Engagement | `likes_count`, `retweets_count`, `comments_count` |
| V1 oracle | `sentiment_score`, `diversity_score`, `context_score` |
| V2 components | `sentiment_impact`, `engagement_score`, `social_capital_v2`, `state_art_score` |
| V3 SCSA-PLUS | `scsa_plus_score`, `author_strength_score`, `recency_score_v3` |
| PCA | `pca1_score`, `pca_variance_explained` |

### Processed `ratings.csv`

| Column | Meaning |
|--------|---------|
| `user_id` | Participant |
| `news_id` | Tweet in the trial |
| `algorithm` | `B1`, `CS`, `SC`, or `SCSA` |
| `position` | Original trial rank |
| `rating` | User rating 1–5 |

Experiments re-rank the same trial items per user — not the full catalog.

Algorithm naming across versions: [Version comparison](VERSION_COMPARISON.md).

---

## Custom algorithm on V3 data

### 1. Build features

```bash
recsocial v3 score    # or: python -m recsocial.cli run v3
```

### 2. Load data

```python
from pathlib import Path
import pandas as pd
from recsocial.shared.pipeline import package_root

root = package_root()
features = pd.read_csv(root / "data/v3/interim/v3_feature_scores.csv")
ratings = pd.read_csv(root / "data/v3/processed/ratings.csv")
features["news_id"] = features["news_id"].astype(str)
ratings["news_id"] = ratings["news_id"].astype(str)
```

### 3. Define a score

```python
features["my_score"] = (
    0.5 * features["scsa_plus_score"]
    + 0.3 * features["engagement_score"]
    + 0.2 * features["pca1_score"]
)
```

### 4. Re-rank trial sessions

```python
from recsocial.shared.reranking import rerank_by_score

rows = []
for (user_id, base_algo), grp in ratings.groupby(["user_id", "algorithm"], sort=False):
    reranked = rerank_by_score(grp, features, "my_score", top_k=10)
    reranked["algorithm"] = f"{base_algo}-MY_ALGO"
    rows.append(reranked[["user_id", "algorithm", "news_id", "ranking", "rating", "score"]])

recommendations = pd.concat(rows, ignore_index=True)
```

Required columns: `user_id`, `algorithm`, `news_id`, `ranking`, `rating`, `score`.

### 5. Evaluate

```python
from recsocial.shared.session_metrics import (
    evaluate_recommendations_by_session,
    settings_from_evaluation_config,
    summarize_session_metrics,
)
from recsocial.slices.v3.config import load_v3_config

cfg = load_v3_config(root / "configs/v3.yaml", base_dir=root)
settings = settings_from_evaluation_config(cfg.evaluation)
detail = evaluate_recommendations_by_session(recommendations, settings)
summary = summarize_session_metrics(detail)
```

Use V3 `evaluation` settings so metrics are comparable to built-in variants.

---

## Add a variant inside the package

1. Add suffix in `configs/v3.yaml` → `rerank_suffixes`
2. Implement in `slices/v3/recommenders.py` using `shared/reranking.py`
3. Run `recsocial v3 experiment` and `recsocial validate`

Config-only tuning: [Improving algorithms](IMPROVING.md).

---

## Quick reference

| Task | Command / path |
|------|----------------|
| Run all | `python -m recsocial.cli run all` |
| Validate | `python -m recsocial.cli validate` |
| Feature matrix | `data/v3/interim/v3_feature_scores.csv` |
| Re-rank helper | `recsocial.shared.reranking.rerank_by_score` |
| Evaluate | `recsocial.shared.session_metrics.evaluate_recommendations_by_session` |
