# V1 Reproduction Plan — Social Capital Recommender

**Paper:** *Exploiting Social Capital for Recommendation in Social Networks* (FedCSIS 2022)  
**Spec:** `SDD - Exploiting Social Capital for Recommendation in Social Networks.md`  
**Scope:** Version 1 of 3 — baseline paper reproduction (Algorithms 1–3, offline evaluation)

---

## 1. Executive Summary

The repository contains **experimental artifacts** from the original research (notebooks, CSV ratings, pre-computed tweet scores) but **does not yet implement a self-contained, reproducible Python pipeline** aligned with the SDD. Scores in `tweets.csv` were produced by an external system (AWS DynamoDB / C# pipeline) and cannot be recomputed from this repo alone.

**V1 goal:** Build `recsocial_py` — a modular Python package that implements the paper's core math, normalizes datasets into the SDD schema, re-runs evaluation, and documents every assumption needed to approach the published metrics (±0.05 tolerance per SDD §18).

| Metric | Paper (SC) | Current `ratingv2.csv` (SC) | Gap |
|--------|------------|----------------------------|-----|
| MRR    | 0.75       | 0.737                      | −0.013 ✓ within tolerance |
| MAP    | 0.62       | 0.705 (V1 eval module)     | +0.085 — legacy script was wrong (0.521) |
| NDCG   | 0.85       | 0.843                      | −0.007 ✓ within tolerance |

V1 must close the **MAP gap** and make the **algorithm path fully traceable** from raw inputs to ranked lists.

---

## 2. Current Repository State

### 2.1 What exists

| Asset | Location | Role |
|-------|----------|------|
| Ground-truth ratings | `database/ratingv2.csv` | 3,100 rows — 73 users × 4 algorithms × 10 items |
| Tweet feature dump | `tweets.csv`, `src/Database/tweets.csv` | 1,373 news items with pre-computed `SocialCapitalScore` |
| Thesis export | `database/thesis-data.csv` | Alternate rating export (71 users, `CS+AS` naming) |
| Evaluation scripts | `results.py`, `results-v2.py` | MRR / MAP / NDCG on `updated_recommendations.csv` |
| Legacy notebooks | `src/MainV1.ipynb` … `MainV3.ipynb` | Exploratory / extended algorithms |
| AWS bridge | `main.py`, `import logging.py` | Fetches scores from DynamoDB `twitter-analytics-v2` |
| SDD specification | `SDD - Exploiting Social Capital…md` | Target architecture & contracts |
| Paper PDF | `Exploiting Social Capital…pdf` | Reference algorithms & results |

### 2.2 Critical gaps

1. **No `recsocial_py` package** — logic is scattered across notebooks and one-off scripts.
2. **No SDD-compliant datasets** — missing `users.csv`, `news.csv`, `comments.csv`, `mentions.csv` in canonical form.
3. **Algorithm not in Python** — `SocialCapitalScore` is read from DynamoDB, not computed via Equations 1–4.
4. **Experiment shape mismatch** — paper: 80 users, 4 rounds × 10 recs = 40 ratings/user; data: ~73 users, mostly 10 ratings/algorithm (1 effective round).
5. **Naming drift** — repo uses `CS` / `SCSA`; paper uses `CS-PLUS` / `SC+SA`.
6. **MAP implementation** — current script may not match paper's AP definition (denominator / relevance handling).
7. **Duplicate & corrupt files** — `updated_recommendationsv3.csv` has broken headers; CSVs duplicated at root and under `src/Database/`.

### 2.3 Version lineage (3 versions)

| Version | Source | Focus |
|---------|--------|-------|
| **V1** | Paper + SDD baseline | Influence, SC, SC+SA, CS-PLUS, B1, prequential eval |
| **V2** | `MainV2.ipynb` | Hybrid TF-IDF features, weighted social capital (`social_capital_score_new`) |
| **V3** | `MainV3.ipynb` | PCA profile vectors, `SCSA_PLUS` / `STATE_ART` re-rankings |

This plan covers **V1 only**. V2/V3 are follow-on milestones after baseline metrics are validated.

---

## 3. Target Directory Layout (V1)

```text
social-capital/
├── docs/
│   ├── V1_REPRODUCTION_PLAN.md      ← this file
│   ├── DATA_INVENTORY.md
│   ├── GAP_ANALYSIS.md
│   └── reproduction_notes.md          ← SDD §23 ambiguities
├── paper/
│   └── Exploiting Social Capital….pdf
├── recsocial_py/
│   ├── pyproject.toml
│   ├── README.md
│   ├── configs/
│   │   ├── default.yaml
│   │   └── reproduction.yaml
│   ├── data/
│   │   ├── raw/                       ← immutable originals (symlinks or copies)
│   │   ├── processed/                   ← SDD-schema CSVs
│   │   └── interim/                     ← scored news, feature matrices
│   ├── notebooks/
│   │   ├── 01_dataset_exploration.ipynb
│   │   ├── 02_algorithm_validation.ipynb
│   │   └── 03_reproduce_results.ipynb
│   ├── src/recsocial/
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── text_features.py
│   │   ├── sentiment.py
│   │   ├── influence.py
│   │   ├── social_capital.py
│   │   ├── user_profile.py
│   │   ├── recommenders.py
│   │   ├── evaluation.py
│   │   ├── experiment.py
│   │   └── reporting.py
│   └── tests/
│       ├── test_influence.py
│       ├── test_social_capital.py
│       ├── test_metrics.py
│       └── fixtures/                  ← synthetic 3-user dataset
├── legacy/                            ← frozen pre-V1 code (see legacy/README.md)
│   ├── notebooks/
│   ├── scripts/
│   └── database/
└── scripts/
    ├── inventory_datasets.py
    └── migrate_to_sdd_schema.py
```

---

## 4. V1 Milestones (aligned with SDD §24)

### Milestone 1 — Data foundation (Week 1)

**Deliverables**

- [ ] `scripts/inventory_datasets.py` — row counts, schema diff, overlap checks
- [ ] `scripts/migrate_to_sdd_schema.py` — transform `tweets.csv` + ratings → SDD files
- [ ] `data/processed/{users,news,comments,mentions,ratings}.csv`
- [ ] `docs/DATA_INVENTORY.md` — auto-generated manifest
- [ ] Archive duplicates into `legacy/`; keep single canonical paths

**Mapping rules (draft)**

| SDD field | Source in `tweets.csv` |
|-----------|------------------------|
| `news_id` | `Id` |
| `author_id` | `AuthorId` |
| `text` | *(needs join — text not in tweets.csv; extract from DynamoDB export or `output_recommendations.csv`)* |
| `likes_count` | `LikeCount` |
| `retweets_count` | `RetweetCount` |
| `comments_count` | `ReplyCount` |
| `mentioned_user_ids` | parse `Mentions` column |

| SDD field | Source for users |
|-----------|------------------|
| `followers_count` | aggregate from tweet author metadata or separate user export |
| `lists_count` | `user_listed_count` from MainV1 raw dataset if available |
| `received_likes_count` | sum of likes on user's tweets |
| `published_news_count` | tweet count per author |

**Blocker to resolve:** Full tweet text + user metadata may require DynamoDB export or locating the raw dataset used in `MainV1.ipynb`.

### Milestone 2 — Core algorithms (Week 2)

**Deliverables**

- [ ] `influence.py` — `PScore`, `RScore`, `IScore` with `strict_equation` and `paper_pseudocode` modes
- [ ] `social_capital.py` — Algorithm 3, recursive comments, mention aggregation
- [ ] `sentiment.py` — pluggable backend (start with `vader`; document Comprehend gap)
- [ ] Unit tests with hand-calculated fixtures

**Validation checkpoints**

- Recompute `SocialCapitalScore` for sample tweets; compare to `tweets.csv` column (expect approximate match depending on mode).
- Verify ranking order for SC on a 20-tweet subset matches original DynamoDB-based ranking.

### Milestone 3 — Text & profiles (Week 3)

**Deliverables**

- [ ] `preprocessing.py` + `text_features.py` — TF-IDF, numeric features, optional PCA
- [ ] `user_profile.py` — incremental update with `rating - 3` weights
- [ ] `recommenders.py` — SC, SC+SA, CS-PLUS, B1, HYBRID

### Milestone 4 — Experiment & evaluation (Week 4)

**Deliverables**

- [ ] `experiment.py` — prequential loop (4 rounds × 10 recs when round metadata available)
- [ ] `evaluation.py` — MRR, P@K, MAP, NDCG (match paper definitions exactly)
- [ ] `reporting.py` — markdown report vs paper Table/Fig targets
- [ ] `03_reproduce_results.ipynb` — end-to-end walkthrough

**Acceptance criteria (SDD §25)**

1. Dataset loads and validates via Pydantic schemas.
2. All four paper algorithms produce ranked lists.
3. Metrics on `ratingv2.csv` reproduced independently of legacy scripts.
4. SC metrics within ±0.05 of paper **or** gaps documented in `reproduction_notes.md`.
5. All unit + integration tests pass.

---

## 5. Algorithm Traceability Matrix

| Paper element | SDD module | Legacy location | V1 status |
|---------------|------------|-----------------|-----------|
| Algorithm 1 — Influence | `influence.py` | DynamoDB (C#) | **To implement** |
| Algorithm 2 — Sentiment | `sentiment.py` | `TweetAnalyzer.py` (BERT, Portuguese) | **Replace with SDD interface** |
| Algorithm 3 — SC score | `social_capital.py` | DynamoDB `SocialCapitalScore` | **To implement** |
| Cosine similarity / CS-PLUS | `user_profile.py`, `recommenders.py` | `MainV2/V3` PCA variants | **V1: baseline only** |
| B1 baseline | `recommenders.py` | Undefined in paper | **Default: interaction_sum** |
| MRR / MAP / NDCG | `evaluation.py` | `results.py` | **Refactor & verify MAP** |
| Prequential eval | `experiment.py` | Not implemented | **To implement** |

---

## 6. Reproducibility Protocol

### 6.1 Frozen inputs

- Copy `database/ratingv2.csv` → `recsocial_py/data/raw/ratings/ratingv2.csv` (checksum recorded)
- Copy `tweets.csv` → `recsocial_py/data/raw/news/tweets_features.csv`
- Never mutate `data/raw/`

### 6.2 Configuration

All paper-sensitive parameters in `configs/reproduction.yaml`:

- `influence.mode: paper_pseudocode`
- `sentiment.backend: vader` (document delta from Amazon Comprehend)
- `recommendation.relevance_threshold: 4`
- `random_seed: 42`

### 6.3 Run sequence

```bash
cd recsocial_py
python -m recsocial migrate --input data/raw --output data/processed
python -m recsocial score --config configs/reproduction.yaml
python -m recsocial experiment --config configs/reproduction.yaml
python -m recsocial report --output reports/v1_baseline
pytest tests/
```

### 6.4 Comparison report

Report must include:

- Dataset statistics (users, news, ratings, rounds)
- Per-algorithm MRR / MAP / NDCG
- Side-by-side with paper values (SDD §18.1–18.3)
- Assumption log from `reproduction_notes.md`

---

## 7. Immediate Actions (Sprint 0)

| # | Action | Owner | Output |
|---|--------|-------|--------|
| 1 | Run `scripts/inventory_datasets.py` | Done in scaffold | `docs/DATA_INVENTORY.md` |
| 2 | Locate full tweet text + user metadata source | Manual | Export or path documented |
| 3 | Move legacy scripts to `legacy/` | Script | Clean root directory |
| 4 | Scaffold `recsocial_py` package | Done in scaffold | Importable `recsocial` module |
| 5 | Implement `evaluation.py` first | Dev | Validate metrics on existing ratings |
| 6 | Fix MAP computation vs paper | Dev | MAP gap analysis in `GAP_ANALYSIS.md` |
| 7 | Write `reproduction_notes.md` | Dev | All SDD §23 items addressed |

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Missing raw tweet text | Cannot recompute TF-IDF / sentiment | Export from DynamoDB or use `output_recommendations.csv` subset |
| DynamoDB unavailable | Cannot verify SC scores | Use `tweets.csv` as reference oracle for V1 validation |
| Round metadata lost | Cannot reproduce 4-round prequential eval | Document single-round limitation; infer rounds from `id` column if possible |
| Sentiment backend differs | SC+SA metrics drift | Ablation with `dummy`, `vader`, stored labels |
| MAP definition ambiguous | False negative on acceptance | Implement both binary and graded AP; compare to paper |

---

## 9. Definition of Done — V1

- [ ] `recsocial_py` installs via `pip install -e .`
- [ ] SDD-schema datasets in `data/processed/`
- [ ] Algorithms 1–3 implemented and unit-tested
- [ ] Four recommenders produce rankings
- [ ] Evaluation report generated with paper comparison
- [ ] Legacy code preserved under `legacy/` with README
- [ ] README updated with V1 quick-start
- [ ] Ready to branch V2 (hybrid features from `MainV2.ipynb`)

---

## 10. References

- Paper PDF: `paper/Exploiting Social Capital for Recommendation in Social Networks.pdf`
- SDD: `SDD - Exploiting Social Capital for Recommendation in Social Networks.md`
- Canonical ratings: `database/ratingv2.csv`
- Canonical news features: `tweets.csv`
