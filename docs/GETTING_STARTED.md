# Getting started

Install the package, run the three paper reproductions, and read the output reports.

## Prerequisites

- Python 3.11+
- Raw data in `recsocial_py/data/raw/`: `ratings.csv`, `tweets.csv`, `users_twitter.csv`

## Install

```bash
cd recsocial_py
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
pytest tests/ -q
recsocial --help
```

## Run

```bash
# All versions (V1 → V2 → V3) + validation
python -m recsocial.cli run all

# Single version
python -m recsocial.cli run v1
python -m recsocial.cli run v2
python -m recsocial.cli run v3
```

Granular steps (`preprocess`, `score`, `experiment`, `plot`) are documented in [Tutorial](TUTORIAL.md#granular-cli).

## Validate

```bash
python -m recsocial.cli validate
```

Writes `reports/validation_summary.md`. Criteria and tolerances: [Evaluation](EVALUATION.md).

```bash
pytest tests/shared/test_paper_validation.py -v
```

## Outputs

| Output | Path |
|--------|------|
| V1 report | `reports/v1/report.md` |
| V2 report | `reports/v2/report.md` |
| V3 report | `reports/v3/report.md` |
| Cross-paper validation | `reports/validation_summary.md` |
| Figures | `reports/v{1,2,3}/figures/` |

Each slice report includes configuration, metrics summary, paper comparison, and a figures index.

## Regenerate figures only

```bash
recsocial v1 plot
recsocial v2 plot
recsocial v3 plot
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: recsocial` | Run from `recsocial_py/` with `pip install -e ".[dev]"` |
| Validation reports no data | Run `recsocial run all` first |
| V2/V3 preprocess errors | Ensure raw CSVs exist in `data/raw/` |

## Next steps

- [Tutorial](TUTORIAL.md) — data layer and custom algorithms on V3 features
- [Version comparison](VERSION_COMPARISON.md) — what differs between V1, V2, V3
- [Improving algorithms](IMPROVING.md) — tune weights and add variants
