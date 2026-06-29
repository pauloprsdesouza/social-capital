# Legacy Archive

Pre-reorganization research artifacts. **Do not extend** — use `recsocial_py/` vertical slices instead.

## Layout

| Path | Contents |
|------|----------|
| `notebooks/` | MainV1, MainV2, MainV3, exploratory Jupyter notebooks |
| `scripts/` | DynamoDB fetchers, legacy `results.py`, ad-hoc `teste*.py`, TweetAnalyzer |
| `database/` | Alternate rating exports (`rating.csv`, `thesis-data.csv`) |
| `outputs/` | Stale recommendation CSVs and duplicate tweet exports |

## Canonical data (active pipelines)

All inputs live under `recsocial_py/data/raw/`:

- `ratings.csv` — trial ratings (was `database/ratingv2.csv`)
- `tweets.csv` — news features + oracle scores
- `users_twitter.csv` — tweet text and engagement metadata (was `output_recommendations.csv`)
- `legacy/v2_recommendations.csv` — V2 comparison baseline
- `legacy/v3_output.csv` — V3 comparison baseline

## Papers and SDD specs

See `docs/papers/`.
