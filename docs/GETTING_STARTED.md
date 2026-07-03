# Getting started

This guide walks through installing the package, running experiments, validating results against the papers, and reading the output reports.

## Prerequisites

- Python 3.11+
- Git clone of this repository
- Raw data in `recsocial_py/data/raw/`:
  - `ratings.csv` — user trial rankings and ratings
  - `tweets.csv` — tweet text and engagement features
  - `users_twitter.csv` — user profiles and follower counts

## 1. Install

```bash
cd recsocial_py
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

Verify:

```bash
pytest tests/ -q
recsocial --help
```

## 2. Run experiments

### All three versions (recommended first run)

Runs V1 → V2 → V3 with shared preprocessing, generates reports, and validates against paper targets:

```bash
python -m recsocial.cli run all
```

Outputs:

| Output | Path |
|--------|------|
| V1 report | `reports/v1/report.md` |
| V2 report | `reports/v2/report.md` |
| V3 report | `reports/v3/report.md` |
| Cross-paper validation | `reports/validation_summary.md` |
| Figures | `reports/v{1,2,3}/figures/` |

### Single version

```bash
python -m recsocial.cli run v1
python -m recsocial.cli run v2
python -m recsocial.cli run v3
```

### Step-by-step (granular control)

```bash
# V1 only
recsocial v1 preprocess
recsocial v1 score
recsocial v1 experiment

# V2 (requires V1 preprocess)
recsocial v1 preprocess
recsocial v2 preprocess
recsocial v2 score
recsocial v2 experiment

# V3 (requires V1 + V2 preprocess)
recsocial v3 preprocess
recsocial v3 score
recsocial v3 experiment
```

## 3. Validate against papers

If reports already exist:

```bash
python -m recsocial.cli validate
```

This compares measured metrics to reference values and writes `reports/validation_summary.md`.

Automated tests:

```bash
pytest tests/shared/test_paper_validation.py tests/v2/test_v2_reference_validation.py -v
```

## 4. Read the reports

Each slice report (`reports/v*/report.md`) contains:

1. **Configuration** — weights and settings used
2. **Metrics summary** — MRR, MAP@10, NDCG@10 per algorithm
3. **Paper comparison** — measured vs published targets (pass/fail)
4. **Figures gallery** — link to `figures/index.md`

The cross-paper summary (`reports/validation_summary.md`) gives a single pass/partial/fail status per version. After `recsocial run all`, all three slices (V1, V2, V3) should show **pass**.

## 5. Regenerate figures only

After changing visualization code, regenerate charts without re-running experiments:

```bash
recsocial v1 plot
recsocial v2 plot
recsocial v3 plot
```

## 6. Typical workflows

### Reproduce paper results

1. `recsocial run all`
2. Open `reports/validation_summary.md`
3. Check per-slice reports for detailed deltas

### Compare your changes

1. Edit config or algorithm code (see [Improving algorithms](IMPROVING.md))
2. `recsocial run v2` (or the slice you changed)
3. `recsocial validate`
4. `pytest tests/`

### Understand differences between versions

Read [Version comparison](VERSION_COMPARISON.md) before diving into slice-specific code.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: recsocial` | Run from `recsocial_py/` with `pip install -e ".[dev]"` |
| Missing report CSVs | Run `recsocial run v1` (or `all`) first |
| V2/V3 fail on preprocess | Ensure V1 raw data exists in `data/raw/` |
| Tests skip on validation | Generate reports with `recsocial run all` |

## Next steps

- [Tutorial: data & algorithms](TUTORIAL.md) — full walkthrough of the CSV data layer, running V1/V2/V3, and using V3 features with your own algorithm
- [Architecture](ARCHITECTURE.md) — how the code is organized
- [Evaluation](EVALUATION.md) — why V1/V2/V3 use different metric protocols
- [Improving algorithms](IMPROVING.md) — where to change scoring and re-ranking
