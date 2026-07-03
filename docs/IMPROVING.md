# Improving the algorithms

Tune, extend, or experiment with the recommender while keeping results reproducible.

## Workflow

1. Baseline: `recsocial run all` && `recsocial validate`
2. Edit config or code (table below)
3. Re-run affected slice: `recsocial run v2` (etc.)
4. `recsocial validate` && `pytest tests/ -q`

## Where to change what

| Goal | Location |
|------|----------|
| V1 Social Capital / influence | `configs/v1.yaml`, `slices/v1/social_capital.py` |
| V1 sentiment | `configs/v1.yaml` → `sentiment`, `slices/v1/sentiment.py` |
| V2 component weights | `configs/v2.yaml` → `social_capital.weights` |
| V2 component formulas | `slices/v2/components.py` |
| V2 STATE_ART | `configs/v2.yaml` → `state_art`, `slices/v2/recommenders.py` |
| V3 SCSA-PLUS | `slices/v3/tweet_metrics.py`, `configs/v3.yaml` |
| V3 PCA | `configs/v3.yaml` → `pca`, `slices/v3/pca_ranking.py` |
| Evaluation protocol | `configs/v*.yaml` → `evaluation` |
| Paper targets / tolerance | `configs/v1.yaml`, `v3.yaml`, `reference_results.yaml` |
| Paired t-tests | `statistics.comparisons` in v2/v3 YAML |

## Example: V2 weights

```yaml
# configs/v2.yaml
social_capital:
  weights:
    sentiment_impact: 0.25
    engagement_score: 0.20
    content_relevance: 0.20
    network_influence: 0.15
    author_influence: 0.10
    content_virality: 0.10
```

```bash
recsocial run v2
recsocial validate
```

## Example: new re-ranking suffix

```yaml
rerank_suffixes:
  my_variant: MY_VARIANT
```

Implement in `slices/v2/recommenders.py` or `slices/v3/recommenders.py` following existing patterns in `shared/reranking.py`.

## Testing

| Scope | Command |
|-------|---------|
| Slice unit tests | `pytest tests/v1/` / `v2/` / `v3/` |
| Paper validation | `pytest tests/shared/test_paper_validation.py` |
| Full suite | `pytest tests/ -q` |

Add unit tests when changing `social_capital.py`, `components.py`, or `tweet_metrics.py`.

## Conventions

1. **Slice isolation** — shared code must not import from slices; V2 must not import V3.
2. **Config first** — tunable values belong in YAML.
3. **Algorithm names** — use `shared/algorithms.py`; variants follow `{BASE}-{SUFFIX}`.
4. **Reports** — write CSV + markdown via existing reporting helpers.

## Code map

| Question | Module |
|----------|--------|
| V1 scoring | `slices/v1/social_capital.py` |
| V2 re-ranking | `slices/v2/recommenders.py`, `shared/reranking.py` |
| V3 PCA | `slices/v3/pca_ranking.py` |
| Metrics | `shared/session_metrics.py` |
| Pipeline | `shared/pipeline.py`, `cli.py` |

See [Architecture](ARCHITECTURE.md) and [Evaluation](EVALUATION.md).

Document intentional deviations in the relevant `docs/v*/reproduction_notes*.md`.
