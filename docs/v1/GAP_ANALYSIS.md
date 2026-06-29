# Gap Analysis — Current Repo vs Paper vs SDD

Generated as part of V1 reproduction planning.

---

## 1. Metrics Comparison

Computed on `database/ratingv2.csv` (73 users, 4 algorithms, 10 items each, relevance ≥ 4).

### MRR

| Algorithm | Paper | Current | Δ | Within ±0.05? |
|-----------|-------|---------|---|---------------|
| SC        | 0.75  | 0.737   | −0.013 | Yes |
| SC+SA (SCSA) | 0.68 | 0.674 | −0.006 | Yes |
| CS-PLUS (CS) | 0.67 | 0.675 | +0.005 | Yes |
| B1        | 0.68  | 0.692   | +0.012 | Yes |

### MAP

**Legacy `results.py` (incorrect denominator):**

| Algorithm | Paper | Legacy script | Δ |
|-----------|-------|---------------|---|
| SC        | 0.62  | 0.521         | −0.099 |
| SC+SA     | 0.53  | 0.414         | −0.116 |
| CS-PLUS   | 0.52  | 0.402         | −0.118 |
| B1        | 0.55  | 0.435         | −0.115 |

**`recsocial.evaluation` (SDD §17.3 — AP / relevant_count):**

| Algorithm | Paper | V1 module | Δ | Within ±0.05? |
|-----------|-------|-----------|---|---------------|
| SC        | 0.62  | 0.705     | +0.085 | No (higher) |
| SC+SA     | 0.53  | 0.634     | +0.104 | No (higher) |
| CS-PLUS   | 0.52  | 0.611     | +0.091 | No (higher) |
| B1        | 0.55  | 0.643     | +0.093 | No (higher) |

The legacy MAP bug (dividing by `min(len, k)` instead of number of relevant items) explained the apparent −0.10 gap. The corrected implementation yields MAP **above** paper values — further investigation needed on whether the paper used a different AP definition or aggregated across rounds differently.

### NDCG@10

| Algorithm | Paper | Current | Δ | Within ±0.05? |
|-----------|-------|---------|---|---------------|
| SC        | 0.85  | 0.843   | −0.007 | Yes |
| SC+SA     | 0.81  | 0.811   | +0.001 | Yes |
| CS-PLUS   | 0.79  | 0.813   | +0.023 | Yes |
| B1        | 0.79  | 0.814   | +0.024 | Yes |

### Interpretation

- **MRR and NDCG** on stored ratings are already close to paper values.
- **MAP in legacy scripts was wrong** — `results.py` used `cum_precision / min(len, k)` instead of dividing by relevant-item count (SDD §17.3).
- **Corrected MAP is slightly above paper values** — may indicate different aggregation (per-round vs per-user-algorithm) or a different AP variant in the original evaluation.
- V1 must document the exact MAP formula used and compare multiple AP variants against the paper.

---

## 2. Algorithm Implementation Gaps

| Component | Paper specification | Current implementation | Gap severity |
|-----------|--------------------|-----------------------|--------------|
| Popularity `PScore(u)` | `1 - exp(-λ·TS)` | Pre-computed in DynamoDB | High — not in Python |
| Reputation `RScore(u)` | `TS/TLS` or `TS` if TLS=0 | Unknown (external) | High |
| Influence `IScore(u)` | `(PScore+TC+TNP)/RScore` + pseudocode fallbacks | Unknown (external) | High |
| News SC | `(TC+TE+STM+STC+TCP)·IScore(u)` × sentiment | `SocialCapitalScore` column | High — black box |
| Sentiment weights | α=1.5, θ=1, β=0.5 | `SentimentScore` float (not labels) | Medium |
| CS-PLUS | Cosine similarity + profile update | `CS` algorithm ratings only | High — ranking logic missing |
| B1 baseline | Not defined in paper | Unknown original definition | Medium |
| Hybrid score | `SC + similarity (>0.7)` | Partial in V2/V3 notebooks | Medium |
| Text preprocessing | Amazon Comprehend pipeline | Portuguese NLTK + BERT in `TweetAnalyzer.py` | High |

---

## 3. Data Schema Gaps

### SDD required files

| File | Status | Notes |
|------|--------|-------|
| `users.csv` | **Missing** | Must derive from tweet author aggregates |
| `news.csv` | **Partial** | `tweets.csv` has metrics but no `text` column |
| `comments.csv` | **Missing** | Reply threads not exported |
| `mentions.csv` | **Partial** | `Mentions` embedded in tweets.csv |
| `ratings.csv` | **Partial** | `ratingv2.csv` — rename columns to SDD format |

### Experiment metadata

| Field | Paper | `ratingv2.csv` |
|-------|-------|----------------|
| Participants | 80 | 73 |
| Rounds | 4 | ~1 (10 items/algo; `id` is not round) |
| Items per round | 10 | 10 |
| Total ratings/user | 40 | 40 (10×4 algorithms, not 4 rounds) |

---

## 4. Code Organization Gaps

| Issue | Examples |
|-------|----------|
| Duplicated CSVs | `tweets.csv` = `src/Database/tweets.csv` |
| Scattered entry points | `main.py`, `teste01-04.py`, `import logging.py` |
| Broken outputs | `updated_recommendationsv3.csv` |
| Hardcoded paths | `social-capital/updated_recommendations.csv` in `results.py` |
| Cloud dependency | `boto3` + DynamoDB for core scores |
| Notebook-only logic | MainV1/V2/V3 not extractable as library |

---

## 5. V1 Priority Fixes

1. **Implement canonical `evaluation.py`** — resolve MAP discrepancy first (validates on existing ratings without reimplementing SC).
2. **Build SDD data migration** — unlock algorithm reimplementation.
3. **Implement `influence.py` + `social_capital.py`** — verify against `tweets.csv` oracle scores.
4. **Document all assumptions** in `reproduction_notes.md`.
5. **Archive legacy** — reduce confusion during V1 development.
