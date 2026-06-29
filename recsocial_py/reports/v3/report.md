# V3 Reproduction Report — SCSA-PLUS

Paper: *Exploiting Social Capital for Improving Personalized Recommendations in Online Social Networks*

Pipeline: SCSA-PLUS social capital + PCA re-ranking (MainV3.ipynb). CSV-only.

## Metrics summary

| Algorithm | MRR | MAP | NDCG |
|-----------|-----|-----|------|
| B1 | 0.748 | 0.687 | 0.758 |
| B1-SCSA_PLUS | 0.762 | 0.676 | 0.778 |
| B1-SCSA_PLUS-SCSA_PLUS_V3 | 0.762 | 0.659 | 0.767 |
| B1-STATE_ART | 0.783 | 0.692 | 0.789 |
| B1-STATE_ART-SCSA_PLUS_V3 | 0.762 | 0.659 | 0.767 |
| CS | 0.732 | 0.656 | 0.734 |
| CS-SCSA_PLUS | 0.619 | 0.584 | 0.702 |
| CS-SCSA_PLUS-SCSA_PLUS_V3 | 0.764 | 0.638 | 0.755 |
| CS-STATE_ART | 0.624 | 0.591 | 0.707 |
| CS-STATE_ART-SCSA_PLUS_V3 | 0.758 | 0.634 | 0.751 |
| SC | 0.792 | 0.761 | 0.800 |
| SC-SCSA_PLUS | 0.774 | 0.705 | 0.795 |
| SC-SCSA_PLUS-SCSA_PLUS_V3 | 0.732 | 0.667 | 0.770 |
| SC-STATE_ART | 0.786 | 0.708 | 0.798 |
| SC-STATE_ART-SCSA_PLUS_V3 | 0.732 | 0.666 | 0.769 |
| SCSA | 0.735 | 0.677 | 0.746 |
| SCSA-SCSA_PLUS | 0.749 | 0.657 | 0.758 |
| SCSA-SCSA_PLUS-SCSA_PLUS_V3 | 0.666 | 0.583 | 0.704 |
| SCSA-STATE_ART | 0.799 | 0.675 | 0.776 |
| SCSA-STATE_ART-SCSA_PLUS_V3 | 0.651 | 0.577 | 0.699 |
| SCSA_PLUS | 0.704 | 0.632 | 0.748 |

## Paired t-tests (p ≤ 0.05 significant)

- **ndcg** SCSA-STATE_ART vs SCSA-SCSA_PLUS: p=0.03721 (n=71)

## Paper targets (§26 — base trial algorithms)

**MRR**: SC=0.793, B1=0.748, SCSA=0.748
**MAP**: SC=0.777, B1=0.728
**NDCG**: SC=0.788, B1=0.753

Note: paper headline **SCSA-PLUS** maps to base algorithm **SC** on stored trial rankings.

## Paper headline vs measured (SC)

- **MRR** measured **0.792** vs paper **0.793** (Δ -0.001)
- **MAP** measured **0.761** vs paper **0.777** (Δ -0.016)
- **NDCG** measured **0.800** vs paper **0.788** (Δ +0.012)

## Legacy output_v3.csv comparison

- B1: MRR 0.748 vs nan | NDCG 0.758 vs nan
- B1-SCSA_PLUS: MRR 0.762 vs 0.6863626201654373 | NDCG 0.778 vs 0.7012714717953418
- B1-SCSA_PLUS-SCSA_PLUS_V3: MRR 0.762 vs 0.6976689976689977 | NDCG 0.767 vs 0.6693947509070166
- B1-STATE_ART: MRR 0.783 vs 0.7349765258215963 | NDCG 0.789 vs 0.8360951116946084
- B1-STATE_ART-SCSA_PLUS_V3: MRR 0.762 vs 0.7176391683433937 | NDCG 0.767 vs 0.819424386468903
- CS: MRR 0.732 vs nan | NDCG 0.734 vs nan
- CS-SCSA_PLUS: MRR 0.619 vs 0.5243787511393146 | NDCG 0.702 vs 0.6258424541144618
- CS-SCSA_PLUS-SCSA_PLUS_V3: MRR 0.764 vs 0.6476663904998621 | NDCG 0.755 vs 0.6379576914016356
- CS-STATE_ART: MRR 0.624 vs 0.5844064386317908 | NDCG 0.707 vs 0.7990423449127787
- CS-STATE_ART-SCSA_PLUS_V3: MRR 0.758 vs 0.6818242790073777 | NDCG 0.751 vs 0.8122721627274306
- SC: MRR 0.792 vs nan | NDCG 0.800 vs nan
- SC-SCSA_PLUS: MRR 0.774 vs 0.7014440644153525 | NDCG 0.795 vs 0.7092883913220218

## Figures

Publication-style charts: [`figures/index.md`](figures/index.md)

Regenerate: `python -m recsocial.cli v3 plot`

## Assumptions

See `docs/v3/reproduction_notes_v3.md`.

## Artifacts

- Features: `C:\Users\paulo\source\repos\social-capital\recsocial_py\data\v3\interim/v3_feature_scores.csv`
- Recommendations: `C:\Users\paulo\source\repos\social-capital\recsocial_py\reports\v3/v3_recommendations.csv`