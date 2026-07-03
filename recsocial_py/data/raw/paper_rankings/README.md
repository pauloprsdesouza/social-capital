# Paper ranking exports

Author-generated recommendation rankings from the original study notebooks.

Used when `reproduction.mode` is `paper_rankings` or `paper_aligned` in `configs/v2.yaml` (default: `paper_aligned`).

| File | Description |
|------|-------------|
| `v2_recommendations.csv` | AMCIS STATE_ART + SCSA_PLUS variants |
| `v2_baseline_recommendations.csv` | Baseline export (STATE_ART variants) |
| `v3_recommendations.csv` | V3 output with PCA suffix variants |

Restore from git history:

```bash
python scripts/restore_paper_rankings.py
```

Modes: `computed` (pipeline only), `paper_rankings` (load these CSVs), `paper_aligned` (default for V2).
