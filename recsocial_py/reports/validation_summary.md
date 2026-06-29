# Cross-Paper Validation Summary

Comparison of implemented algorithms against published reference values.

## Overview

| Slice | Paper | Checks | Strict pass | Relaxed pass | Status | Report |
|-------|-------|--------|-------------|--------------|--------|--------|
| V1 | FedCSIS 2022 — Social Capital Recommender | 12 | 12 | 12 | **pass** | [report](reports/v1/report.md) |
| V2 | AMCIS 2024 — Unlocking the Power of Social Capital | 96 | 53 | 73 | **partial** | [report](reports/v2/report.md) |
| V3 | SCSA-PLUS — Enhanced Personalized Recommendations | 7 | 6 | 6 | **partial** | [report](reports/v3/report.md) |

**Total:** 71/115 strict, 91/115 relaxed (± slice-specific tolerances).

## V1 — FedCSIS trial metrics

| Algorithm | Metric | Expected | Actual | Δ | Status |
|-----------|--------|----------|--------|---|--------|
| B1 | MRR | 0.680 | 0.692 | +0.012 | pass |
| B1 | MAP | 0.550 | 0.551 | +0.001 | pass |
| B1 | NDCG | 0.790 | 0.814 | +0.024 | pass |
| CS-PLUS | MRR | 0.670 | 0.675 | +0.005 | pass |
| CS-PLUS | MAP | 0.520 | 0.515 | -0.005 | pass |
| CS-PLUS | NDCG | 0.790 | 0.813 | +0.023 | pass |
| SC | MRR | 0.750 | 0.737 | -0.013 | pass |
| SC | MAP | 0.620 | 0.624 | +0.004 | pass |
| SC | NDCG | 0.850 | 0.843 | -0.007 | pass |
| SC+SA | MRR | 0.680 | 0.674 | -0.006 | pass |
| SC+SA | MAP | 0.530 | 0.527 | -0.003 | pass |
| SC+SA | NDCG | 0.810 | 0.811 | +0.001 | pass |

## V2 — AMCIS Figures 3–10

V2 uses chart-aligned MAP and binary NDCG. Full tables: `reports/v2/tables/`.

Largest relaxed-tolerance gaps (sample):

| Figure | Algorithm | Metric | Expected | Actual | Δ |
|--------|-----------|--------|----------|--------|---|
| figure_3_b1 | B1-SCSA_PLUS | MRR | 0.793 | 0.740 | -0.053 |
| figure_3_b1 | B1-SCSA_PLUS | MAP_10 | 0.777 | 0.689 | -0.088 |
| figure_3_b1 | B1-SCSA_PLUS_V3 | MRR | 0.665 | 0.720 | +0.055 |
| figure_3_b1 | B1-SCSA_PLUS_V3 | NDCG_10 | 0.659 | 0.768 | +0.109 |
| figure_4_cs | CS-SCSA_PLUS | MAP_10 | 0.669 | 0.593 | -0.076 |
| figure_4_cs | CS-STATE_ART | MRR | 0.680 | 0.589 | -0.091 |
| figure_4_cs | CS-STATE_ART | NDCG_10 | 0.817 | 0.707 | -0.110 |
| figure_4_cs | CS-SCSA_PLUS_V3 | NDCG_10 | 0.660 | 0.724 | +0.064 |
| figure_5_sc | SC-SCSA_PLUS | MAP_10 | 0.786 | 0.714 | -0.072 |
| figure_5_sc | SC-SCSA_PLUS_V3 | NDCG_10 | 0.676 | 0.768 | +0.092 |

## V3 — SCSA-PLUS §26 headline metrics

| Algorithm | Metric | Expected | Actual | Δ | Status |
|-----------|--------|----------|--------|---|--------|
| SC | MRR | 0.793 | 0.792 | -0.001 | pass |
| SC | MAP | 0.777 | 0.761 | -0.016 | pass |
| SC | NDCG | 0.788 | 0.800 | +0.012 | pass |
| B1 | MRR | 0.748 | 0.748 | +0.000 | pass |
| B1 | MAP | 0.728 | 0.687 | -0.041 | fail |
| B1 | NDCG | 0.753 | 0.758 | +0.005 | pass |
| SCSA | MRR | 0.748 | 0.735 | -0.013 | pass |

## Commands

```bash
# Run all slices and regenerate reports
python -m recsocial.cli run all

# Validate only (requires existing report CSVs)
python -m recsocial.cli validate
```
