# V2 Reproduction Notes

Assumptions for **Unlocking the Power of Social Capital** (AMCIS 2024) implementation.

## 1. STATE_ART baseline

Paper cites Tiwari et al. (2021) but does not ship code.  
**V2 default:** `STATE_ART = content_relevance + engagement_score` (SDD §12.3).

## 2. SCSA_PLUS vs SCSA_PLUS_V3

- **SCSA_PLUS_V3:** weighted component formula (SI, ES, CR, NI, AI, CV) from SDD §10 / `MainV2.ipynb`.
- **SCSA_PLUS:** V1 paper SC+SA score (`recsocial` influence-based engine) applied as re-ranker.

## 3. Re-ranking protocol

For each base algorithm (B1, CS, SC, SCSA) and user session:

1. **{BASE}-STATE_ART** — preserve original trial order from `ratingv2.csv`.
2. **{BASE}-SCSA_PLUS** — re-rank the same 10 trial items by V1 SC+SA score.
3. **{BASE}-SCSA_PLUS_V3** — re-rank by enhanced component Social Capital score.

## 4. Scaling

Default: `minmax_0_1` in `configs/v2.yaml` (matches `MainV2.ipynb`).

- `sentiment_impact` from scaled sentiment
- `engagement_score` = mean of scaled interaction counts
- `content_relevance` = mean TF-IDF vector
- `network_influence` = mentions + urls counts
- `author_influence` = mentions count (`paper_compatible` mode)
- `content_virality` = scaled retweets + quotes

Alternative: `scaling_mode: standard` for ablations.

## 5. Extended formula

Recency, diversity, context weights default to 0. Enable via `use_extended_formula: true`.
