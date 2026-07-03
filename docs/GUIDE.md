# User guide

Install, run, validate, use the data layer, build custom algorithms, and tune the recommender.

All inputs and outputs are CSV files under `recsocial_py/data/` and `recsocial_py/reports/` (no database).

---

## Install

**Prerequisites:** Python 3.11+, raw data in `recsocial_py/data/raw/` (`ratings.csv`, `tweets.csv`, `users_twitter.csv`).

```bash
cd recsocial_py
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
pytest tests/ -q
recsocial --help
```

## Run and validate

```bash
# All versions (V1 → V2 → V3) + validation
python -m recsocial.cli run all

# Single version
python -m recsocial.cli run v1
python -m recsocial.cli run v2
python -m recsocial.cli run v3

# Validate only (requires existing reports)
python -m recsocial.cli validate
pytest tests/shared/test_paper_validation.py -v
```

Validation criteria: [Evaluation](EVALUATION.md).

### Outputs

| Output | Path |
|--------|------|
| Slice reports | `reports/v{1,2,3}/report.md` |
| Validation summary | `reports/validation_summary.md` |
| Figures | `reports/v{1,2,3}/figures/` |

Regenerate figures without re-running experiments: `recsocial v1 plot`, `recsocial v2 plot`, `recsocial v3 plot`.

### CLI reference

| Command | Description |
|---------|-------------|
| `recsocial run all` | Full pipeline + validation |
| `recsocial run v1\|v2\|v3` | Single slice |
| `recsocial validate` | Compare reports to paper targets |
| `recsocial v1 preprocess\|score\|experiment\|plot\|report` | V1 granular steps |
| `recsocial v2 preprocess\|score\|experiment\|plot` | V2 granular steps |
| `recsocial v3 preprocess\|score\|experiment\|plot` | V3 granular steps |

Granular pipeline:

```bash
recsocial v1 preprocess && recsocial v1 score && recsocial v1 experiment
recsocial v2 preprocess && recsocial v2 score && recsocial v2 experiment
recsocial v3 preprocess && recsocial v3 score && recsocial v3 experiment
```

### Python API

```python
from recsocial.shared.pipeline import package_root, run_all_pipelines, run_v3_pipeline
from recsocial.shared.paper_validation import validate_all_papers

root = package_root()
run_all_pipelines(root)
validate_all_papers(root)
```

### Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: recsocial` | `pip install -e ".[dev]"` from `recsocial_py/` |
| Validation reports no data | Run `recsocial run all` first |
| V2/V3 preprocess errors | Ensure raw CSVs exist in `data/raw/` |

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

Join key: `news_id` (cast to string).

### `v3_feature_scores.csv` columns

| Group | Examples |
|-------|----------|
| Identity | `news_id`, `author_id`, `text` |
| Engagement | `likes_count`, `retweets_count`, `comments_count` |
| V1 oracle | `sentiment_score`, `diversity_score`, `context_score` |
| V2 components | `sentiment_impact`, `engagement_score`, `social_capital_v2` |
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

Experiments re-rank the same trial items per user. Algorithm naming: [Versions](VERSIONS.md).

---

## Custom algorithm (V3)

### 1. Build features

```bash
recsocial v3 score
```

### 2. Load, score, re-rank, evaluate

```python
from pathlib import Path
import pandas as pd
from recsocial.shared.pipeline import package_root
from recsocial.shared.reranking import rerank_by_score
from recsocial.shared.session_metrics import (
    evaluate_recommendations_by_session,
    settings_from_evaluation_config,
    summarize_session_metrics,
)
from recsocial.slices.v3.config import load_v3_config

root = package_root()
features = pd.read_csv(root / "data/v3/interim/v3_feature_scores.csv")
ratings = pd.read_csv(root / "data/v3/processed/ratings.csv")
features["news_id"] = features["news_id"].astype(str)
ratings["news_id"] = ratings["news_id"].astype(str)

features["my_score"] = (
    0.5 * features["scsa_plus_score"]
    + 0.3 * features["engagement_score"]
    + 0.2 * features["pca1_score"]
)

rows = []
for (user_id, base_algo), grp in ratings.groupby(["user_id", "algorithm"], sort=False):
    reranked = rerank_by_score(grp, features, "my_score", top_k=10)
    reranked["algorithm"] = f"{base_algo}-MY_ALGO"
    rows.append(reranked[["user_id", "algorithm", "news_id", "ranking", "rating", "score"]])

recommendations = pd.concat(rows, ignore_index=True)

cfg = load_v3_config(root / "configs/v3.yaml", base_dir=root)
detail = evaluate_recommendations_by_session(
    recommendations, settings_from_evaluation_config(cfg.evaluation)
)
summary = summarize_session_metrics(detail)
```

Required recommendation columns: `user_id`, `algorithm`, `news_id`, `ranking`, `rating`, `score`.

---

## Tune and extend

### Workflow

1. `recsocial run all` && `recsocial validate`
2. Edit config or code (table below)
3. `recsocial run v2` (or affected slice)
4. `recsocial validate` && `pytest tests/ -q`

### Where to change what

| Goal | Location |
|------|----------|
| V1 Social Capital / influence | `configs/v1.yaml`, `slices/v1/social_capital.py` |
| V1 sentiment | `configs/v1.yaml` → `sentiment` |
| V2 component weights | `configs/v2.yaml` → `social_capital.weights` |
| V2 component formulas | `slices/v2/components.py` |
| V2 STATE_ART | `configs/v2.yaml` → `state_art` |
| V3 SCSA-PLUS | `slices/v3/tweet_metrics.py` |
| V3 PCA | `configs/v3.yaml` → `pca` |
| Evaluation protocol | `configs/v*.yaml` → `evaluation` |
| Paper targets / tolerance | `configs/v1.yaml`, `v3.yaml`, `reference_results.yaml` |
| New rerank suffix | `rerank_suffixes` in YAML + `slices/v*/recommenders.py` |

### Conventions

1. Shared code must not import from slices; V2 must not import V3.
2. Tunable values belong in YAML.
3. Algorithm names: `shared/algorithms.py`; variants follow `{BASE}-{SUFFIX}`.

See [Architecture](ARCHITECTURE.md) for module map and [Versions](VERSIONS.md) to document implementation assumptions.
