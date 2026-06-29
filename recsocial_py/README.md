# recsocial_py

Python package for Social Capital recommender reproduction (V1 / V2 / V3 vertical slices).

## Quick start

```bash
pip install -e ".[dev]"

# Run all three papers end-to-end (preprocess → score → experiment → reports → validation)
python -m recsocial.cli run all

# Or use the convenience script
python scripts/run_all.py

# Validate existing reports against paper reference values
python -m recsocial.cli validate
```

Reports: `reports/v1/report.md`, `reports/v2/report.md`, `reports/v3/report.md`  
Cross-paper validation: `reports/validation_summary.md`

## Structure

```text
configs/          v1.yaml, v2.yaml, v3.yaml, reference_results.yaml (V2 targets)
data/raw/         ratings.csv, tweets.csv, users_twitter.csv
data/v1|v2|v3/    Processed + interim CSVs per slice
reports/v1|v2|v3/ Metrics, figures, and report.md per slice
reports/          validation_summary.md (cross-paper validation)
scripts/          run_all.py, run_v1.py, run_v2.py, run_v3.py
src/recsocial/
  cli.py          recsocial run|validate|v1|v2|v3
  shared/         evaluation, pipeline, paper_validation, reference_validation
  slices/v1/      FedCSIS 2022 baseline
  slices/v2/      AMCIS 2024 enhanced components + re-ranking
  slices/v3/      SCSA-PLUS + PCA + paired t-tests
tests/            Unit and paper-validation tests
```

## Commands

| Command | Description |
|---------|-------------|
| `recsocial run all` | V1 → V2 → V3 full pipeline + validation |
| `recsocial run v1\|v2\|v3` | Single slice pipeline |
| `recsocial run v1 --validate` | Run V1 then validate all slices |
| `recsocial validate` | Compare reports to paper targets |
| `recsocial v1 experiment` | Individual step (also v2, v3) |
| `recsocial v1 plot` | Regenerate figures from saved CSVs |

## Paper validation

Each slice is validated against its paper reference values:

| Slice | Paper | Reference source | Tolerance |
|-------|-------|------------------|-----------|
| V1 | FedCSIS 2022 | `configs/v1.yaml` → `paper_targets` | ±0.05 |
| V2 | AMCIS 2024 | `configs/reference_results.yaml` (Figs 3–10) | ±0.03 strict / ±0.05 relaxed |
| V3 | SCSA-PLUS §26 | `configs/v3.yaml` → `paper_targets` | ±0.02 |

```bash
pytest tests/shared/test_paper_validation.py tests/v2/test_v2_reference_validation.py
```

## Architecture

```text
shared/                 Cross-cutting, slice-agnostic
  pipeline.py           End-to-end run orchestration
  paper_validation.py   Unified V1/V2/V3 paper comparison
  reference_validation.py  V2 figure-level AMCIS validation
  experiment_runner.py  Standard evaluate → persist → report flow

slices/v1|v2|v3/        Vertical slices — domain logic only
```

See [../docs/v1/reproduction_notes.md](../docs/v1/reproduction_notes.md) and sibling v2/v3 docs for assumptions.
