# V1 Reproduction Report

CSV-only pipeline — no database.

## Trial metrics (stored user ratings)

| Algorithm | MRR | MAP | NDCG |
|-----------|-----|-----|------|
| B1 | 0.692 | 0.551 | 0.814 |
| CS-PLUS | 0.675 | 0.515 | 0.813 |
| SC | 0.737 | 0.624 | 0.843 |
| SC+SA | 0.674 | 0.527 | 0.811 |

## Comparison vs paper targets

| Algorithm | Metric | Measured | Paper | Delta | Status |
|-----------|--------|----------|-------|-------|--------|
| B1 | MRR | 0.692 | 0.68 | +0.012 | pass |
| B1 | MAP | 0.551 | 0.55 | +0.001 | pass |
| B1 | NDCG | 0.814 | 0.79 | +0.024 | pass |
| CS-PLUS | MRR | 0.675 | 0.67 | +0.005 | pass |
| CS-PLUS | MAP | 0.515 | 0.52 | -0.005 | pass |
| CS-PLUS | NDCG | 0.813 | 0.79 | +0.023 | pass |
| SC | MRR | 0.737 | 0.75 | -0.013 | pass |
| SC | MAP | 0.624 | 0.62 | +0.004 | pass |
| SC | NDCG | 0.843 | 0.85 | -0.007 | pass |
| SC+SA | MRR | 0.674 | 0.68 | -0.006 | pass |
| SC+SA | MAP | 0.527 | 0.53 | -0.003 | pass |
| SC+SA | NDCG | 0.811 | 0.81 | +0.001 | pass |

## Oracle SC score validation

- **pearson_corr**: 0.7803
- **spearman_rank_corr**: 0.8918
- **mean_abs_error**: 815369.2673
- **median_abs_error**: 11709.0560

## Figures

Publication-style charts: [`figures/index.md`](figures/index.md)

Regenerate without re-running the experiment:

```bash
python -m recsocial.cli v1 plot
```

## Assumptions

See `docs/v1/reproduction_notes.md` for ambiguities and V1 defaults.

## Data artifacts

- Processed CSVs: `C:\Users\paulo\source\repos\social-capital\recsocial_py\data\v1\processed`
- Interim scores: `C:\Users\paulo\source\repos\social-capital\recsocial_py\data\v1\interim`
- Reports: `C:\Users\paulo\source\repos\social-capital\recsocial_py\reports\v1`