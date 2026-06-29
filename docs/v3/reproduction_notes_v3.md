# V3 Reproduction Notes

Assumptions for **Exploiting Social Capital for Improving Personalized Recommendations in Online Social Networks** (SCSA-PLUS + PCA).

## 1. SCSA-PLUS social capital (SDD §18)

Composite score per tweet:

```text
(author_strength + interactions + media + diversity + mentions_strength + token_length + context) × recency
```

- **author_strength:** reputation × influence (§11–13)
- **reputation:** listed-count fallback when mention/reply texts are unavailable
- **influence:** log(followers+1) × (engagement_rate + 1), optional impressions
- **recency:** logarithmic decay (§14)
- **context:** TF-IDF cosine vs topic keywords when oracle ContextScore is zero
- **diversity:** oracle DiversityScore from `tweets.csv`

## 2. PCA re-ranking (MainV3.ipynb)

1. Build hybrid matrix: numeric metrics + TF-IDF(tokens + hashtags + text)
2. `StandardScaler` → `PCA(95% variance)`
3. **pca1_score** = first principal component
4. For each algorithm in V2 recommendations, append `-SCSA_PLUS_V3` and re-rank by `pca1_score`

Output CSV columns match the original V3 notebook export (base algorithms + PCA variants).

## 3. Re-ranking protocol

| Algorithm | Description |
|-----------|-------------|
| `{BASE}-STATE_ART` | Original trial order from `ratingv2.csv` |
| `{BASE}-SCSA_PLUS` | Re-rank trial items by `scsa_plus_score` |
| `{algo}-SCSA_PLUS_V3` | PCA re-rank of each row above |
| `SCSA_PLUS` | Standalone per-user SCSA-PLUS ranking (t-tests vs B1/SCSA) |

## 4. Evaluation

- Relevance threshold: rating ≥ 4
- Metrics: MRR, MAP@10, NDCG@10, Precision@1–5
- Paired t-tests (SDD §23): B1 vs SCSA_PLUS, SCSA vs SCSA_PLUS, STATE_ART vs *-SCSA_PLUS

## 5. Known gaps vs paper targets

Paper targets (§26): SCSA-PLUS MRR ≈ 0.793, MAP ≈ 0.777, NDCG ≈ 0.788.

Measured values depend on reputation fallback, PCA feature set, and trial-only re-ranking (not full candidate pool). See `recsocial_py/reports/v3/report.md` and `recsocial_py/reports/validation_summary.md` for current numbers.

> **Note:** In V2, the suffix `SCSA_PLUS_V3` re-ranks by V2 component scores. In V3, the same suffix name means PCA re-ranking. See [Version comparison](../VERSION_COMPARISON.md).

## 6. Run

```bash
cd recsocial_py
pip install -e ".[dev]"
python -m recsocial.cli run v3
```

Or step-by-step: `python -m recsocial.cli v3 experiment --config configs/v3.yaml`
