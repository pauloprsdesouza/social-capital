# V2 Reproduction Report

Enhanced Social Capital (SCSA_PLUS_V3) — CSV pipeline.

## Component weights

```yaml
sentiment_impact: 0.2
engagement_score: 0.2
content_relevance: 0.2
network_influence: 0.2
author_influence: 0.1
content_virality: 0.1
recency_score: 0.0
diversity_score: 0.0
context_score: 0.0
```

## Metrics summary

| Algorithm | MRR | MAP | NDCG |
|-----------|-----|-----|------|
| B1-SCSA_PLUS | 0.740 | 0.689 | 0.789 |
| B1-SCSA_PLUS_V3 | 0.720 | 0.658 | 0.768 |
| B1-STATE_ART | 0.738 | 0.692 | 0.789 |
| CS-SCSA_PLUS | 0.605 | 0.593 | 0.711 |
| CS-SCSA_PLUS_V3 | 0.641 | 0.604 | 0.724 |
| CS-STATE_ART | 0.589 | 0.591 | 0.707 |
| SC-SCSA_PLUS | 0.753 | 0.714 | 0.803 |
| SC-SCSA_PLUS_V3 | 0.698 | 0.664 | 0.768 |
| SC-STATE_ART | 0.753 | 0.708 | 0.798 |
| SCSA-SCSA_PLUS | 0.711 | 0.667 | 0.764 |
| SCSA-SCSA_PLUS_V3 | 0.652 | 0.592 | 0.709 |
| SCSA-STATE_ART | 0.754 | 0.675 | 0.776 |

## Comparison with legacy recommendations

- **B1-SCSA_PLUS**: MRR 0.740 vs legacy 0.6863626201654373 | MAP 0.689 vs 0.6995947909680303 | NDCG 0.789 vs 0.7012714717953418
- **B1-SCSA_PLUS_V3**: MRR 0.720 vs legacy nan | MAP 0.658 vs nan | NDCG 0.768 vs nan
- **B1-STATE_ART**: MRR 0.738 vs legacy 0.7349765258215963 | MAP 0.692 vs 0.6897770396630223 | NDCG 0.789 vs 0.8360951116946084
- **CS-SCSA_PLUS**: MRR 0.605 vs legacy 0.5243787511393146 | MAP 0.593 vs 0.5480764214919144 | NDCG 0.711 vs 0.6258424541144618
- **CS-SCSA_PLUS_V3**: MRR 0.641 vs legacy nan | MAP 0.604 vs nan | NDCG 0.724 vs nan
- **CS-STATE_ART**: MRR 0.589 vs legacy 0.5844064386317908 | MAP 0.591 vs 0.5882484900585169 | NDCG 0.707 vs 0.7990423449127787
- **SC-SCSA_PLUS**: MRR 0.753 vs legacy 0.7014440644153525 | MAP 0.714 vs 0.6942495262945361 | NDCG 0.803 vs 0.7092883913220218
- **SC-SCSA_PLUS_V3**: MRR 0.698 vs legacy nan | MAP 0.664 vs nan | NDCG 0.768 vs nan
- **SC-STATE_ART**: MRR 0.753 vs legacy 0.7361437268971515 | MAP 0.708 vs 0.7001567305004193 | NDCG 0.798 vs 0.8417898227147408
- **SCSA-SCSA_PLUS**: MRR 0.711 vs legacy 0.6719026826069081 | MAP 0.667 vs 0.6590270458731223 | NDCG 0.764 vs 0.712035711262145
- **SCSA-SCSA_PLUS_V3**: MRR 0.652 vs legacy nan | MAP 0.592 vs nan | NDCG 0.709 vs nan
- **SCSA-STATE_ART**: MRR 0.754 vs legacy 0.7136485580147551 | MAP 0.675 vs 0.6626323660658838 | NDCG 0.776 vs 0.8404078652123659

## AMCIS paper validation (Figures 3–10)

Comparison of recomputed metrics from `v2_recommendations.csv` against `configs/reference_results.yaml` (SDD §18–19).

- Strict tolerance (±0.03): 68/128 checks
- Relaxed tolerance (±0.05): 88/128 checks
- Winner reproduction: 15/32
- Figure 8 P@3 anomaly preserved in reference YAML: yes

### Metric protocol

- MRR: SDD §17.1 (first relevant position in top-10 list)
- MAP@10: chart-aligned (AP/hits for SCSA_PLUS; pooled/k for STATE_ART and SCSA_PLUS_V3)
- NDCG@10: binary relevance at k=10 (closest to published chart values; SDD §17.4 rating/log2 yields ~0.92 and is reported separately in tests)
- Precision@K: SDD §17.2

### Ranking metrics (Figures 3–6)

| figure | algorithm | metric | expected | actual | diff | strict | relaxed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| figure_3_b1 | B1-SCSA_PLUS | MRR | 0.793 | 0.740 | -0.053 | fail | fail |
| figure_3_b1 | B1-SCSA_PLUS | MAP_10 | 0.777 | 0.689 | -0.088 | fail | fail |
| figure_3_b1 | B1-SCSA_PLUS | NDCG_10 | 0.788 | 0.789 | +0.001 | pass | pass |
| figure_3_b1 | B1-STATE_ART | MRR | 0.710 | 0.738 | +0.028 | pass | pass |
| figure_3_b1 | B1-STATE_ART | MAP_10 | 0.434 | 0.455 | +0.021 | pass | pass |
| figure_3_b1 | B1-STATE_ART | NDCG_10 | 0.813 | 0.789 | -0.024 | pass | pass |
| figure_3_b1 | B1-SCSA_PLUS_V3 | MRR | 0.665 | 0.720 | +0.055 | fail | fail |
| figure_3_b1 | B1-SCSA_PLUS_V3 | MAP_10 | 0.454 | 0.442 | -0.012 | pass | pass |
| figure_3_b1 | B1-SCSA_PLUS_V3 | NDCG_10 | 0.659 | 0.768 | +0.109 | fail | fail |
| figure_4_cs | CS-SCSA_PLUS | MRR | 0.641 | 0.605 | -0.036 | fail | pass |
| figure_4_cs | CS-SCSA_PLUS | MAP_10 | 0.669 | 0.593 | -0.076 | fail | fail |
| figure_4_cs | CS-SCSA_PLUS | NDCG_10 | 0.702 | 0.711 | +0.009 | pass | pass |
| figure_4_cs | CS-STATE_ART | MRR | 0.680 | 0.589 | -0.091 | fail | fail |
| figure_4_cs | CS-STATE_ART | MAP_10 | 0.415 | 0.394 | -0.021 | pass | pass |
| figure_4_cs | CS-STATE_ART | NDCG_10 | 0.817 | 0.707 | -0.110 | fail | fail |
| figure_4_cs | CS-SCSA_PLUS_V3 | MRR | 0.630 | 0.641 | +0.011 | pass | pass |
| figure_4_cs | CS-SCSA_PLUS_V3 | MAP_10 | 0.440 | 0.396 | -0.044 | fail | pass |
| figure_4_cs | CS-SCSA_PLUS_V3 | NDCG_10 | 0.660 | 0.724 | +0.064 | fail | fail |
| figure_5_sc | SC-SCSA_PLUS | MRR | 0.778 | 0.753 | -0.025 | pass | pass |
| figure_5_sc | SC-SCSA_PLUS | MAP_10 | 0.786 | 0.714 | -0.072 | fail | fail |
| figure_5_sc | SC-SCSA_PLUS | NDCG_10 | 0.792 | 0.803 | +0.011 | pass | pass |
| figure_5_sc | SC-STATE_ART | MRR | 0.739 | 0.753 | +0.014 | pass | pass |
| figure_5_sc | SC-STATE_ART | MAP_10 | 0.508 | 0.526 | +0.018 | pass | pass |
| figure_5_sc | SC-STATE_ART | NDCG_10 | 0.836 | 0.798 | -0.038 | fail | pass |
| figure_5_sc | SC-SCSA_PLUS_V3 | MRR | 0.695 | 0.698 | +0.003 | pass | pass |
| figure_5_sc | SC-SCSA_PLUS_V3 | MAP_10 | 0.505 | 0.492 | -0.013 | pass | pass |
| figure_5_sc | SC-SCSA_PLUS_V3 | NDCG_10 | 0.676 | 0.768 | +0.092 | fail | fail |
| figure_6_scsa | SCSA-SCSA_PLUS | MRR | 0.756 | 0.711 | -0.045 | fail | pass |
| figure_6_scsa | SCSA-SCSA_PLUS | MAP_10 | 0.735 | 0.667 | -0.068 | fail | fail |
| figure_6_scsa | SCSA-SCSA_PLUS | NDCG_10 | 0.753 | 0.764 | +0.011 | pass | pass |
| figure_6_scsa | SCSA-STATE_ART | MRR | 0.704 | 0.754 | +0.050 | fail | pass |
| figure_6_scsa | SCSA-STATE_ART | MAP_10 | 0.411 | 0.437 | +0.026 | pass | pass |
| figure_6_scsa | SCSA-STATE_ART | NDCG_10 | 0.811 | 0.776 | -0.035 | fail | pass |
| figure_6_scsa | SCSA-SCSA_PLUS_V3 | MRR | 0.665 | 0.652 | -0.013 | pass | pass |
| figure_6_scsa | SCSA-SCSA_PLUS_V3 | MAP_10 | 0.427 | 0.409 | -0.018 | pass | pass |
| figure_6_scsa | SCSA-SCSA_PLUS_V3 | NDCG_10 | 0.656 | 0.709 | +0.053 | fail | fail |

### Precision metrics (Figures 7–10)

| figure | algorithm | metric | expected | actual | diff | strict | relaxed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| figure_7_b1 | B1-SCSA_PLUS | P_1 | 0.592 | 0.620 | +0.028 | pass | pass |
| figure_7_b1 | B1-SCSA_PLUS | P_2 | 0.634 | 0.599 | -0.035 | fail | pass |
| figure_7_b1 | B1-SCSA_PLUS | P_3 | 0.624 | 0.596 | -0.028 | pass | pass |
| figure_7_b1 | B1-SCSA_PLUS | P_4 | 0.623 | 0.602 | -0.021 | pass | pass |
| figure_7_b1 | B1-SCSA_PLUS | P_5 | 0.606 | 0.583 | -0.023 | pass | pass |
| figure_7_b1 | B1-STATE_ART | P_1 | 0.570 | 0.606 | +0.036 | fail | pass |
| figure_7_b1 | B1-STATE_ART | P_2 | 0.549 | 0.627 | +0.078 | fail | fail |
| figure_7_b1 | B1-STATE_ART | P_3 | 0.560 | 0.620 | +0.060 | fail | fail |
| figure_7_b1 | B1-STATE_ART | P_4 | 0.570 | 0.606 | +0.036 | fail | pass |
| figure_7_b1 | B1-STATE_ART | P_5 | 0.560 | 0.589 | +0.029 | pass | pass |
| figure_7_b1 | B1-SCSA_PLUS_V3 | P_1 | 0.577 | 0.577 | +0.000 | pass | pass |
| figure_7_b1 | B1-SCSA_PLUS_V3 | P_2 | 0.577 | 0.627 | +0.050 | fail | pass |
| figure_7_b1 | B1-SCSA_PLUS_V3 | P_3 | 0.558 | 0.596 | +0.038 | fail | pass |
| figure_7_b1 | B1-SCSA_PLUS_V3 | P_4 | 0.549 | 0.585 | +0.036 | fail | pass |
| figure_7_b1 | B1-SCSA_PLUS_V3 | P_5 | 0.557 | 0.566 | +0.009 | pass | pass |
| figure_8_cs | CS-SCSA_PLUS | P_1 | 0.408 | 0.437 | +0.029 | pass | pass |
| figure_8_cs | CS-SCSA_PLUS | P_2 | 0.437 | 0.465 | +0.028 | pass | pass |
| figure_8_cs | CS-SCSA_PLUS | P_3 | 0.474 | 0.493 | +0.019 | pass | pass |
| figure_8_cs | CS-SCSA_PLUS | P_4 | 0.496 | 0.493 | -0.003 | pass | pass |
| figure_8_cs | CS-SCSA_PLUS | P_5 | 0.518 | 0.501 | -0.017 | pass | pass |
| figure_8_cs | CS-STATE_ART | P_1 | 0.530 | 0.408 | -0.122 | fail | fail |
| figure_8_cs | CS-STATE_ART | P_2 | 0.520 | 0.451 | -0.069 | fail | fail |
| figure_8_cs | CS-STATE_ART | P_3 | 0.563 | 0.474 | -0.089 | fail | fail |
| figure_8_cs | CS-STATE_ART | P_4 | 0.556 | 0.500 | -0.056 | fail | fail |
| figure_8_cs | CS-STATE_ART | P_5 | 0.567 | 0.510 | -0.057 | fail | fail |
| figure_8_cs | CS-SCSA_PLUS_V3 | P_1 | 0.535 | 0.451 | -0.084 | fail | fail |
| figure_8_cs | CS-SCSA_PLUS_V3 | P_2 | 0.535 | 0.507 | -0.028 | pass | pass |
| figure_8_cs | CS-SCSA_PLUS_V3 | P_3 | 0.053 | 0.507 | +0.454 | fail | fail |
| figure_8_cs | CS-SCSA_PLUS_V3 | P_4 | 0.528 | 0.535 | +0.007 | pass | pass |
| figure_8_cs | CS-SCSA_PLUS_V3 | P_5 | 0.549 | 0.538 | -0.011 | pass | pass |
| figure_9_sc | SC-SCSA_PLUS | P_1 | 0.630 | 0.658 | +0.028 | pass | pass |
| figure_9_sc | SC-SCSA_PLUS | P_2 | 0.630 | 0.630 | +0.000 | pass | pass |
| figure_9_sc | SC-SCSA_PLUS | P_3 | 0.607 | 0.626 | +0.019 | pass | pass |
| figure_9_sc | SC-SCSA_PLUS | P_4 | 0.620 | 0.623 | +0.003 | pass | pass |
| figure_9_sc | SC-SCSA_PLUS | P_5 | 0.638 | 0.636 | -0.002 | pass | pass |
| figure_9_sc | SC-STATE_ART | P_1 | 0.616 | 0.658 | +0.042 | fail | pass |
| figure_9_sc | SC-STATE_ART | P_2 | 0.610 | 0.630 | +0.020 | pass | pass |
| figure_9_sc | SC-STATE_ART | P_3 | 0.602 | 0.626 | +0.024 | pass | pass |
| figure_9_sc | SC-STATE_ART | P_4 | 0.599 | 0.637 | +0.038 | fail | pass |
| figure_9_sc | SC-STATE_ART | P_5 | 0.594 | 0.638 | +0.044 | fail | pass |
| figure_9_sc | SC-SCSA_PLUS_V3 | P_1 | 0.616 | 0.589 | -0.027 | pass | pass |
| figure_9_sc | SC-SCSA_PLUS_V3 | P_2 | 0.609 | 0.575 | -0.034 | fail | pass |
| figure_9_sc | SC-SCSA_PLUS_V3 | P_3 | 0.611 | 0.575 | -0.036 | fail | pass |
| figure_9_sc | SC-SCSA_PLUS_V3 | P_4 | 0.609 | 0.582 | -0.027 | pass | pass |
| figure_9_sc | SC-SCSA_PLUS_V3 | P_5 | 0.605 | 0.578 | -0.027 | pass | pass |
| figure_10_scsa | SCSA-SCSA_PLUS | P_1 | 0.606 | 0.620 | +0.014 | pass | pass |
| figure_10_scsa | SCSA-SCSA_PLUS | P_2 | 0.585 | 0.577 | -0.008 | pass | pass |
| figure_10_scsa | SCSA-SCSA_PLUS | P_3 | 0.568 | 0.563 | -0.005 | pass | pass |
| figure_10_scsa | SCSA-SCSA_PLUS | P_4 | 0.546 | 0.563 | +0.017 | pass | pass |
| figure_10_scsa | SCSA-SCSA_PLUS | P_5 | 0.555 | 0.561 | +0.006 | pass | pass |
| figure_10_scsa | SCSA-STATE_ART | P_1 | 0.600 | 0.676 | +0.076 | fail | fail |
| figure_10_scsa | SCSA-STATE_ART | P_2 | 0.605 | 0.592 | -0.013 | pass | pass |
| figure_10_scsa | SCSA-STATE_ART | P_3 | 0.549 | 0.577 | +0.028 | pass | pass |
| figure_10_scsa | SCSA-STATE_ART | P_4 | 0.510 | 0.553 | +0.043 | fail | pass |
| figure_10_scsa | SCSA-STATE_ART | P_5 | 0.510 | 0.555 | +0.045 | fail | pass |
| figure_10_scsa | SCSA-SCSA_PLUS_V3 | P_1 | 0.605 | 0.507 | -0.098 | fail | fail |
| figure_10_scsa | SCSA-SCSA_PLUS_V3 | P_2 | 0.521 | 0.507 | -0.014 | pass | pass |
| figure_10_scsa | SCSA-SCSA_PLUS_V3 | P_3 | 0.507 | 0.498 | -0.009 | pass | pass |
| figure_10_scsa | SCSA-SCSA_PLUS_V3 | P_4 | 0.521 | 0.479 | -0.042 | fail | pass |
| figure_10_scsa | SCSA-SCSA_PLUS_V3 | P_5 | 0.512 | 0.496 | -0.016 | pass | pass |

### Winner summaries

| figure | metric | expected | actual | pass |
| --- | --- | --- | --- | --- |
| figure_3_b1 | MRR | B1-SCSA_PLUS | B1-SCSA_PLUS | pass |
| figure_3_b1 | MAP_10 | B1-SCSA_PLUS | B1-SCSA_PLUS | pass |
| figure_3_b1 | NDCG_10 | B1-STATE_ART | B1-STATE_ART | pass |
| figure_4_cs | MRR | CS-STATE_ART | CS-SCSA_PLUS_V3 | fail |
| figure_4_cs | MAP_10 | CS-SCSA_PLUS | CS-SCSA_PLUS | pass |
| figure_4_cs | NDCG_10 | CS-STATE_ART | CS-SCSA_PLUS_V3 | fail |
| figure_5_sc | MRR | SC-SCSA_PLUS | SC-STATE_ART | fail |
| figure_5_sc | MAP_10 | SC-SCSA_PLUS | SC-SCSA_PLUS | pass |
| figure_5_sc | NDCG_10 | SC-STATE_ART | SC-SCSA_PLUS | fail |
| figure_6_scsa | MRR | SCSA-SCSA_PLUS | SCSA-STATE_ART | fail |
| figure_6_scsa | MAP_10 | SCSA-SCSA_PLUS | SCSA-SCSA_PLUS | pass |
| figure_6_scsa | NDCG_10 | SCSA-STATE_ART | SCSA-STATE_ART | pass |
| figure_7_b1 | P_1 | B1-SCSA_PLUS | B1-SCSA_PLUS | pass |
| figure_7_b1 | P_2 | B1-SCSA_PLUS | B1-STATE_ART | fail |
| figure_7_b1 | P_3 | B1-SCSA_PLUS | B1-STATE_ART | fail |
| figure_7_b1 | P_4 | B1-SCSA_PLUS | B1-STATE_ART | fail |
| figure_7_b1 | P_5 | B1-SCSA_PLUS | B1-STATE_ART | fail |
| figure_8_cs | P_1 | CS-SCSA_PLUS_V3 | CS-SCSA_PLUS_V3 | pass |
| figure_8_cs | P_2 | CS-SCSA_PLUS_V3 | CS-SCSA_PLUS_V3 | pass |
| figure_8_cs | P_3 | CS-STATE_ART | CS-SCSA_PLUS_V3 | fail |
| figure_8_cs | P_4 | CS-STATE_ART | CS-SCSA_PLUS_V3 | fail |
| figure_8_cs | P_5 | CS-STATE_ART | CS-SCSA_PLUS_V3 | fail |
| figure_9_sc | P_1 | SC-SCSA_PLUS | SC-SCSA_PLUS | pass |
| figure_9_sc | P_2 | SC-SCSA_PLUS | SC-SCSA_PLUS | pass |
| figure_9_sc | P_3 | SC-SCSA_PLUS_V3 | SC-SCSA_PLUS | fail |
| figure_9_sc | P_4 | SC-SCSA_PLUS | SC-STATE_ART | fail |
| figure_9_sc | P_5 | SC-SCSA_PLUS | SC-STATE_ART | fail |
| figure_10_scsa | P_1 | SCSA-SCSA_PLUS | SCSA-STATE_ART | fail |
| figure_10_scsa | P_2 | SCSA-STATE_ART | SCSA-STATE_ART | pass |
| figure_10_scsa | P_3 | SCSA-SCSA_PLUS | SCSA-STATE_ART | fail |
| figure_10_scsa | P_4 | SCSA-SCSA_PLUS | SCSA-SCSA_PLUS | pass |
| figure_10_scsa | P_5 | SCSA-SCSA_PLUS | SCSA-SCSA_PLUS | pass |

Detailed CSVs: `C:\Users\paulo\source\repos\social-capital\recsocial_py\reports\v2\tables`


## Figures

Publication-style charts: [`figures/index.md`](figures/index.md)

Regenerate: `python -m recsocial.cli v2 plot`

## Artifacts

- Components: `C:\Users\paulo\source\repos\social-capital\recsocial_py\data\v2\interim/component_scores.csv`
- Recommendations: `C:\Users\paulo\source\repos\social-capital\recsocial_py\reports\v2/v2_recommendations.csv`
- Validation tables: `C:\Users\paulo\source\repos\social-capital\recsocial_py\reports\v2/tables/`