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

## 4. Scaling (legacy_notebook mode)

Matches `MainV2.ipynb`:

- `StandardScaler` on sentiment → `sentiment_impact`
- Interaction counts scaled in-place, then `engagement_score = mean(likes, retweets, replies, quotes)`
- `content_relevance = mean(TF-IDF vector)`
- `network_influence = mentions_count + urls_count` (raw counts)
- `author_influence = mentions_count` (paper_compatible mode)
- `content_virality = scaled_retweets + scaled_quotes`

SDD `minmax_0_1` mode available via config for ablations.

## 5. Extended formula

Recency, diversity, context weights default to 0. Enable via `use_extended_formula: true`.

## 6. Legacy validation

Compare output to `updated_recommendations.csv` (STATE_ART + SCSA_PLUS variants without V3 suffix).
