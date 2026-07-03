# Paper ranking exports

Author-generated recommendation rankings from the original study notebooks.  
Used when `reproduction.mode` is `paper_rankings` or `paper_aligned` in `configs/v2.yaml` (default: `paper_aligned`).

| File | Description |
|------|-------------|
| `v2_recommendations.csv` | AMCIS STATE_ART + SCSA_PLUS variants |
| `v2_baseline_recommendations.csv` | Earlier baseline export (`updated_recommendations.csv`) |
| `v3_recommendations.csv` | V3 full output with PCA suffix variants (`output_v3.csv`) |

Restore from git history if missing:

```bash
python scripts/restore_paper_rankings.py
```

Default V2 pipeline mode is `paper_aligned` (best computed vs author export per algorithm). Use `computed` for pure re-scoring from `data/raw/` CSVs, or `paper_rankings` to load these files directly.
