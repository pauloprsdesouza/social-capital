# Evaluation and validation

Metrics, evaluation protocols, paper reference values, and validation output.

## Metrics

All versions report offline metrics at **relevance threshold ≥ 4**:

| Metric | Meaning |
|--------|---------|
| **MRR** | Reciprocal rank of the first relevant item |
| **MAP@10** | Mean average precision over top 10 |
| **NDCG@10** | Normalized discounted cumulative gain at k=10 |
| **Precision@K** | Relevant fraction in top K (K = 1…5, 10) |

Implementation: `recsocial.shared.evaluation`

## Protocols by version

Configured in each slice YAML under `evaluation`:

| Setting | V1 | V2 | V3 |
|---------|----|----|-----|
| `metric_protocol` | `session_list` | `session_list` | `paper_notebook` |
| `map_protocol` | `fedcsis_pooled` | `sdd` | `sdd` |
| `graded_ndcg` | `true` | `false` | `true` |
| `relevance_threshold` | 4 | 4 | 4 |
| `ndcg_k` | 10 | 10 | 10 |

- **session_list** — sorted rank column per `(user_id, algorithm)` (V1, V2)
- **paper_notebook** — notebook rank-column semantics (V3); `shared/paper_metrics.py`
- **fedcsis_pooled** — FedCSIS MAP pooling (V1 only)

### V2 figure validation

`reference_validation.py` uses chart-aligned conventions for AMCIS Figures 3–10:

- **NDCG@10:** binary relevance
- **MAP@10:** AP/hits for SCSA_PLUS; pooled/k for STATE_ART and SCSA_PLUS_V3

Component scaling: `minmax_0_1` in `configs/v2.yaml`.

## Paper reference values

| Version | Config | Content |
|---------|--------|---------|
| V1 | `configs/v1.yaml` → `paper_targets` | MRR, MAP, NDCG for SC, SC+SA, CS-PLUS, B1 |
| V2 | `configs/reference_results.yaml` | Figures 3–10 ranking + precision + winners |
| V3 | `configs/v3.yaml` → `paper_targets` | §26 headline metrics (`aliases.SCSA_PLUS → SC`) |

### Tolerances

| Version | Tolerance | Checks |
|---------|-----------|--------|
| V1 | ±0.05 | 12 |
| V2 | ±0.11 relaxed | 96 |
| V3 | ±0.05 | 7 |

V2 config excerpt:

```yaml
validation:
  algorithmic_reproduction_tolerance: 0.03
  relaxed_reproduction_tolerance: 0.11
```

## Reproduction modes (V2)

| Mode | Behavior |
|------|----------|
| `computed` | Score from `data/raw/` and re-rank trials |
| `paper_rankings` | Load author CSVs from `data/raw/paper_rankings/` |
| `paper_aligned` | **Default** — per algorithm, use computed or author rows per config |

See `data/raw/paper_rankings/README.md`.

## Running validation

```bash
recsocial validate
pytest tests/shared/test_paper_validation.py
pytest tests/v2/test_v2_reference_validation.py
```

| Output | Path |
|--------|------|
| Summary | `reports/validation_summary.md` |
| Details | `reports/validation_details.csv` |
| V2 tables | `reports/v2/tables/` |

## Validation status

| Status | Definition |
|--------|------------|
| **pass** | All checks within configured tolerance |
| **fail** | One or more checks outside tolerance |

Run `recsocial validate` after `recsocial run all` for current status.

## Algorithm labels

Internal codes map to paper labels via `recsocial.shared.algorithms`:

| Internal | Paper label |
|----------|-------------|
| `B1` | B1 |
| `CS` | CS-PLUS |
| `SC` | SC |
| `SCSA` | SC+SA |

V2/V3 add suffixes: `B1-STATE_ART`, `SC-SCSA_PLUS`, `SC-SCSA_PLUS_V3`, etc.

## V1 oracle validation

Computed Social Capital scores vs oracle values in `tweets.csv`:

- `reports/v1/oracle_validation.csv`
- Figure: `reports/v1/figures/fig04_oracle_validation.png`

## Statistical tests (V2, V3)

Paired t-tests from `statistics.comparisons` in each YAML.

Outputs: `reports/v2/paired_ttests.csv`, `reports/v3/paired_ttests.csv`

V3 also produces `correlation_matrix.csv` and `ranking_shifts.csv`.
