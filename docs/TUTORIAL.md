# Tutorial: data, algorithms, and custom experiments

This guide is a hands-on walkthrough for two common tasks:

1. **Run all algorithm versions** (V1, V2, V3) and read the results.
2. **Use the V3 feature data** (the project’s CSV “database”) with your own ranking algorithm.

There is **no SQL database** in this project. All inputs and outputs are **CSV files** under `recsocial_py/data/` and `recsocial_py/reports/`. The sections below refer to this file-based data as the **data layer**.

---

## Prerequisites

```bash
cd recsocial_py
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

Raw source files must exist in `data/raw/`:

| File | Contents |
|------|----------|
| `ratings.csv` | User trial sessions: which items each baseline showed, in what order, with ratings |
| `tweets.csv` | Tweet text, engagement counts, oracle scores |
| `users_twitter.csv` | User profiles, follower counts, listed counts |

---

## Part 1 — Run all algorithm versions

### One command (recommended)

Runs V1 → V2 → V3 with shared preprocessing, writes reports, and validates against paper targets:

```bash
python -m recsocial.cli run all
```

### Run a single version

```bash
python -m recsocial.cli run v1   # FedCSIS 2022 baseline
python -m recsocial.cli run v2   # AMCIS 2024 enhanced components
python -m recsocial.cli run v3   # SCSA-PLUS + PCA
```

### Step-by-step (granular control)

Use this when you want to inspect intermediate CSVs between stages.

```bash
# --- V1 ---
recsocial v1 preprocess    # raw → data/v1/processed/
recsocial v1 score         # compute SC / SC+SA scores
recsocial v1 experiment    # run trial experiment → reports/v1/

# --- V2 (needs V1 preprocess) ---
recsocial v1 preprocess
recsocial v2 preprocess    # enrich news → data/v2/processed/
recsocial v2 score         # component scores → data/v2/interim/
recsocial v2 experiment    # re-rank trials → reports/v2/

# --- V3 (needs V1 + V2 preprocess) ---
recsocial v3 preprocess
recsocial v3 score         # SCSA-PLUS + PCA → data/v3/
recsocial v3 experiment    # re-rank + statistics → reports/v3/
```

### Validate against papers

After reports exist:

```bash
python -m recsocial.cli validate
```

Output: `reports/validation_summary.md` (pass / partial / fail per version).

### Python API (same as CLI)

```python
from recsocial.shared.pipeline import package_root, run_all_pipelines, run_v1_pipeline, run_v2_pipeline, run_v3_pipeline

root = package_root()

# All versions
paths = run_all_pipelines(root)

# Or one version
v3_paths = run_v3_pipeline(root)
for name, path in v3_paths.items():
    print(f"{name}: {path}")
```

### Where to read results

| Version | Report | Recommendations CSV | Metrics summary |
|---------|--------|---------------------|-----------------|
| V1 | `reports/v1/report.md` | `reports/v1/rerank_recommendations.csv` | `reports/v1/trial_metrics_summary.csv` |
| V2 | `reports/v2/report.md` | `reports/v2/v2_recommendations.csv` | `reports/v2/v2_metrics_summary.csv` |
| V3 | `reports/v3/report.md` | `reports/v3/v3_recommendations.csv` | `reports/v3/v3_metrics_summary.csv` |

Cross-version validation: `reports/validation_summary.md`

---

## Part 2 — Understand the data layer

### Pipeline overview

```text
data/raw/                          ← source data (committed to git)
    │
    ▼  V1 preprocess
data/v1/processed/                 ← users, news, ratings, comments, mentions
data/v1/interim/                   ← scored_news_sc.csv, scored_news_scsa.csv
    │
    ▼  V2 preprocess + score
data/v2/processed/                 ← news_enriched.csv, users.csv, ratings.csv
data/v2/interim/                   ← component_scores.csv
    │
    ▼  V3 score
data/v3/processed/                 ← ratings.csv, user_strength.csv
data/v3/interim/                   ← v3_feature_scores.csv  ★ main feature matrix
    │
    ▼  experiments
reports/v1|v2|v3/                  ← recommendations, metrics, figures, report.md
```

Generated paths (`data/v1/interim/`, `data/v2/`, `data/v3/`, `reports/`) are **gitignored**. Regenerate anytime with `recsocial run all`.

### Key files for external / custom use

| Path | When to use it |
|------|----------------|
| `data/raw/ratings.csv` | Original trial format from the paper dataset |
| `data/v3/processed/ratings.csv` | Normalized trial sessions (user, algorithm, news_id, position, rating) |
| `data/v3/interim/v3_feature_scores.csv` | **One row per tweet** — all V2 + V3 features and scores |
| `data/v3/processed/user_strength.csv` | Per-user reputation and influence |
| `data/v2/interim/component_scores.csv` | V2 component breakdown only (no V3 / PCA) |
| `reports/v3/v3_recommendations.csv` | Built-in V3 algorithm outputs (all variants) |

### `v3_feature_scores.csv` — column guide

Each row is one tweet (`news_id`). Important columns:

| Column group | Examples | Meaning |
|--------------|----------|---------|
| Identity | `news_id`, `author_id`, `text` | Tweet and author |
| Engagement | `likes_count`, `retweets_count`, `comments_count`, `impression_count` | Raw counts |
| V1 oracle | `sentiment_score`, `diversity_score`, `context_score` | Precomputed paper scores |
| V2 components | `sentiment_impact`, `engagement_score`, `content_relevance`, `network_influence`, `author_influence`, `content_virality`, `social_capital_v2`, `state_art_score` | AMCIS weighted components |
| V3 SCSA-PLUS | `author_strength_score`, `mentions_strength_score`, `scsa_plus_score`, `recency_score_v3` | SCSA-PLUS formula inputs and output |
| User strength | `reputation_score`, `influence_score_v3`, `user_strength_score` | Author-level metrics |
| PCA | `pca1_score`, `pca_variance_explained`, `pca_n_components` | First principal component for re-ranking |

**Join key:** always `news_id` (cast to string on both sides).

### `ratings.csv` (processed) — column guide

| Column | Meaning |
|--------|---------|
| `user_id` | Participant |
| `news_id` | Tweet shown in the trial |
| `algorithm` | Base algorithm: `B1`, `CS`, `SC`, or `SCSA` |
| `position` | Original rank in the trial (1 = top) |
| `rating` | User rating (1–5); used as ground truth for metrics |
| `session_id`, `round_id` | Trial metadata |

The experiment **re-ranks the same trial items** for each user — it does not retrieve new candidates from the full catalog.

---

## Part 3 — Algorithms in each version

Four **base algorithms** appear in every version:

| Code | Paper label | Idea |
|------|---------------|------|
| `B1` | B1 | Popularity / engagement baseline |
| `CS` | CS-PLUS | Content similarity (user profile vs tweet TF-IDF) |
| `SC` | SC | Social Capital (influence + popularity + reputation) |
| `SCSA` | SC+SA | Social Capital + VADER sentiment |

### V1 outputs

V1 replays trials and evaluates the base algorithms plus re-ranking variants. See `reports/v1/trial_metrics_summary.csv`.

### V2 re-ranking variants

For each base algorithm `BASE`, V2 produces:

| Algorithm name | Meaning |
|----------------|---------|
| `{BASE}-STATE_ART` | Keep original trial order |
| `{BASE}-SCSA_PLUS` | Re-rank trial items by V1 SC+SA score |
| `{BASE}-SCSA_PLUS_V3` | Re-rank by **V2 weighted component score** |

> In V2, the suffix `SCSA_PLUS_V3` means V2 component re-ranking — **not** V3 PCA. See [Version comparison](VERSION_COMPARISON.md).

### V3 re-ranking variants

| Algorithm name | Meaning |
|----------------|---------|
| `{BASE}-STATE_ART` | Original trial order |
| `{BASE}-SCSA_PLUS` | Re-rank by `scsa_plus_score` |
| `{BASE}-SCSA_PLUS_V3` | PCA first-component re-rank (`pca1_score`) |
| `SCSA_PLUS` | Standalone per-user SCSA-PLUS ranking |

Canonical naming helpers: `recsocial.shared.algorithms`

---

## Part 4 — Use V3 data with your own algorithm

This section shows how to load the V3 feature matrix, apply a custom score, re-rank trial items, and evaluate — **without modifying the package**.

### Step 1 — Build the feature data

If `data/v3/interim/v3_feature_scores.csv` does not exist yet:

```bash
python -m recsocial.cli run v3
# or only the scoring step:
recsocial v3 score
```

### Step 2 — Load data

Paths below assume your working directory is `recsocial_py/`. From the repo root, prefix paths with `recsocial_py/` or use `package_root()`:

```python
from pathlib import Path

import pandas as pd
from recsocial.shared.pipeline import package_root

root = package_root()  # always resolves to recsocial_py/

features = pd.read_csv(root / "data/v3/interim/v3_feature_scores.csv")
ratings = pd.read_csv(root / "data/v3/processed/ratings.csv")

features["news_id"] = features["news_id"].astype(str)
ratings["news_id"] = ratings["news_id"].astype(str)
```

### Step 3 — Define your score

Add a column to `features` with your algorithm’s score for each tweet:

```python
def my_score(row: pd.Series) -> float:
    return (
        0.5 * row["scsa_plus_score"]
        + 0.3 * row["engagement_score"]
        + 0.2 * row["pca1_score"]
    )

features["my_score"] = features.apply(my_score, axis=1)
```

You can also use any combination of columns, train a model on features, or load scores from an external system — as long as you end up with `news_id` + a numeric score column.

### Step 4 — Re-rank trial sessions

The paper protocol re-ranks **the same items** each user already saw in a trial:

```python
from recsocial.shared.reranking import rerank_by_score

rows = []
for (user_id, base_algo), grp in ratings.groupby(["user_id", "algorithm"], sort=False):
    reranked = rerank_by_score(grp, features, "my_score", top_k=10)
    reranked["algorithm"] = f"{base_algo}-MY_ALGO"
    rows.append(
        reranked[["user_id", "algorithm", "news_id", "ranking", "rating", "score"]]
    )

recommendations = pd.concat(rows, ignore_index=True)
recommendations.to_csv("my_recommendations.csv", index=False)
```

Required output columns: `user_id`, `algorithm`, `news_id`, `ranking`, `rating`, `score`.

### Step 5 — Evaluate

Use the same metric settings as V3 so results are comparable:

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

print(summary)  # mean MRR, MAP, NDCG per algorithm
detail.to_csv("my_metrics_detail.csv", index=False)
```

### Alternative — rank from the full item pool

If your algorithm ranks **all tweets** (not just re-ranking a fixed trial), use `v3_feature_scores.csv` as the candidate pool:

```python
def rank_full_pool(user_id: str, features: pd.DataFrame, score_col: str, top_k: int = 10):
    pool = features.nlargest(top_k, score_col).copy()
    pool["user_id"] = user_id
    pool["algorithm"] = "MY_ALGO"
    pool["ranking"] = range(1, len(pool) + 1)
    pool["score"] = pool[score_col]
    # Attach ratings where available (0 if item was not rated in trials)
    return pool[["user_id", "algorithm", "news_id", "ranking", "rating", "score"]]
```

**Note:** Full-pool ranking is valid for your own experiments, but numbers will **not** match the published V3 paper, which uses trial re-ranking only.

---

## Part 5 — Plug a custom algorithm into the package

To run alongside built-in variants and get standard reports:

1. **Add a suffix** in `configs/v3.yaml`:

```yaml
rerank_suffixes:
  state_art: STATE_ART
  scsa_plus: SCSA_PLUS
  scsa_plus_v3: SCSA_PLUS_V3
  my_variant: MY_VARIANT
```

2. **Implement scoring** in `src/recsocial/slices/v3/recommenders.py` — follow the `build_v3_recommendations` pattern and use helpers from `shared/reranking.py` (`rerank_by_score`, `append_suffix_rerank`, `build_trial_rerank_bundle`).

3. **Add t-test comparisons** (optional) in `configs/v3.yaml` → `statistics.comparisons`.

4. **Run and validate:**

```bash
recsocial v3 experiment
recsocial validate
pytest tests/ -q
```

For config-only tuning (weights, thresholds), see [Improving algorithms](IMPROVING.md).

---

## Part 6 — Compare your algorithm to built-in versions

After running the built-in pipeline:

```python
import pandas as pd

builtin = pd.read_csv("recsocial_py/reports/v3/v3_metrics_summary.csv")
mine = pd.read_csv("my_metrics_detail.csv")  # or aggregate your detail to summary

# Compare mean MRR for your variant vs SCSA_PLUS
print(builtin[["algorithm", "mrr", "map", "ndcg"]])
print(mine.groupby("algorithm")[["mrr", "map", "ndcg"]].mean())
```

Built-in V3 headline targets (from `configs/v3.yaml`):

| Metric | SCSA_PLUS (SC alias) target |
|--------|------------------------------|
| MRR | 0.793 |
| MAP | 0.777 |
| NDCG | 0.788 |

Tolerance: ±0.02. Live status: `recsocial validate`.

---

## Quick reference

| Task | Command / file |
|------|----------------|
| Run everything | `python -m recsocial.cli run all` |
| Run V3 only | `python -m recsocial.cli run v3` |
| Build V3 features only | `recsocial v3 score` |
| Main feature matrix | `data/v3/interim/v3_feature_scores.csv` |
| Trial ground truth | `data/v3/processed/ratings.csv` |
| Built-in V3 results | `reports/v3/v3_recommendations.csv` |
| Re-rank helper | `recsocial.shared.reranking.rerank_by_score` |
| Evaluate sessions | `recsocial.shared.session_metrics.evaluate_recommendations_by_session` |
| Full pipeline (Python) | `recsocial.shared.pipeline.run_all_pipelines` |

## Next steps

- [Getting started](GETTING_STARTED.md) — install, troubleshoot, regenerate figures
- [Version comparison](VERSION_COMPARISON.md) — what differs between V1 / V2 / V3
- [Evaluation](EVALUATION.md) — metric protocols and paper targets
- [Improving algorithms](IMPROVING.md) — tune weights and add variants inside the package
- [Architecture](ARCHITECTURE.md) — module map and data flow diagram
