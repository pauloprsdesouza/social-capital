# Reproduction Notes — Known Ambiguities

Per SDD §23. Every assumption used in V1 must be recorded here.

---

## 1. Dataset schema

**Ambiguity:** Paper does not publish full dataset schema.  
**V1 assumption:** Derive SDD schema from `data/raw/tweets.csv` and `data/raw/ratings.csv`. Tweet text is taken from the raw export when available.

## 2. Text preprocessing

**Ambiguity:** Exact tokenization/stopword rules not specified.  
**V1 assumption:** English pipeline per SDD §9 (`lowercase`, `remove_urls`, `remove_stopwords`, `ngram_range: [1,2]`).

## 3. User profile update

**Ambiguity:** Internal vector update formula not fully specified.  
**V1 assumption:** `rating_weight = rating - 3`; profile = weighted average of rated news TF-IDF+numeric vectors after each round.

## 4. B1 baseline

**Ambiguity:** Paper names B1 but does not define computation.  
**V1 assumption:** `B1 = likes_count + retweets_count + comments_count` (SDD §15.5 default). Will report sensitivity to `chronological` and `popularity` baselines.

## 5. CS-PLUS implementation

**Ambiguity:** Exact feature vector and PCA settings not specified.  
**V1 assumption:** Hybrid matrix (TF-IDF + normalized interaction counts + author metrics); cosine similarity; cold-start falls back to SC-only (SDD §14.3).

## 6. Sentiment labels

**Ambiguity:** Paper used Amazon Comprehend; labels not stored in dataset.  
**V1 assumption:** VADER backend with SDD default weights (positive=1.5, negative=1.0, neutral=0.5, mixed=0.5). Ablation against `dummy` (neutral=1.0) to measure backend impact.

## 7. Influence score — equation vs pseudocode

**Ambiguity:** Equations 1–3 differ from Algorithm 1 fallbacks.  
**V1 assumption:** Default `paper_pseudocode` mode (zero followers → max_followers, zero lists → β=1, verified → +θ=1). Run `strict_equation` as ablation.

## 8. Sentiment weight tuning

**Ambiguity:** α, θ, β were empirically chosen; tuning process not described.  
**V1 assumption:** Use paper values (1.5, 1.0, 0.5). Grid search documented as optional experiment.

## 9. Hybrid score scale

**Ambiguity:** Raw SC may dominate cosine similarity.  
**V1 assumption:** Implement both `raw_hybrid` and `normalized_hybrid`; report which is closer to paper metrics.

## 10. MAP metric definition

**Ambiguity:** Multiple MAP conventions appear in early project scripts.  
**V1 assumption:** SDD §17.3 — AP = mean of P@k at each relevant position; MAP = mean AP across user-algorithm groups.

## 11. Oracle SocialCapitalScore

**Ambiguity:** `tweets.csv` oracle scores were produced by the original C#/DynamoDB pipeline with additional features.  
**V1 assumption:** Paper formula (Algorithms 1–3) implemented in Python; oracle scores used for correlation validation only.

## 12. Experiment rounds

**Ambiguity:** `ratingv2.csv` `id` column is not `round_id` (values 1–376, one per user-algorithm).  
**V1 assumption:** Treat each user-algorithm group as one evaluation session (10 items). Full 4-round prequential replay requires round metadata recovery or re-simulation.

## 13. Algorithm naming

| Paper | Repo `ratingv2.csv` | V1 canonical name |
|-------|---------------------|-------------------|
| SC | SC | SC |
| SC+SA | SCSA | SC+SA |
| CS-PLUS | CS | CS-PLUS |
| B1 | B1 | B1 |
