# Social Capital Recommender Reproduction

Three vertical slices (V1 → V2 → V3) reproducing the Social Capital recommender papers. CSV-only — no database or cloud dependencies.

| Slice | Paper | Run |
|-------|-------|-----|
| **V1** | [FedCSIS 2022 baseline](docs/papers/Exploiting%20Social%20Capital%20for%20Recommendation%20in%20Social%20Networks.pdf) | `python recsocial_py/scripts/run_v1.py` |
| **V2** | [AMCIS 2024 enhanced SC](docs/papers/Unlocking%20the%20Power%20of%20Social%20Capital%20Advanced%20Strategies%20for%20Enhanced%20Personalized%20Recommendations%20in%20Online%20Social%20Networks.pdf) | `python recsocial_py/scripts/run_v2.py` |
| **V3** | [SCSA-PLUS + PCA](docs/papers/Exploiting%20Social%20Capital%20for%20Improving%20Personalized%20Recommendations%20in%20Online%20Social%20Networks.pdf) | `python recsocial_py/scripts/run_v3.py` |

## Quick start

```bash
cd recsocial_py
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
pytest tests/
python scripts/run_v1.py        # or run_v2.py / run_v3.py
```

## Repository layout (vertical slices)

```text
recsocial_py/                     # Python package — all active code
  configs/v1.yaml, v2.yaml, v3.yaml
  data/raw/                       # Canonical inputs (ratings, tweets, users)
  data/v1|v2|v3/processed|interim/
  reports/v1|v2|v3/
  scripts/run_v1.py, run_v2.py, run_v3.py
  src/recsocial/
    shared/                       # evaluation, utils, schemas (cross-cutting)
    slices/v1/                    # V1: migrate, score, recommend, experiment
    slices/v2/                    # V2: components, re-ranking
    slices/v3/                    # V3: SCSA-PLUS, PCA, statistics
  tests/v1|v2|v3/

docs/
  v1/  v2/  v3/                   # Reproduction notes per slice
  papers/                         # SDD markdown + PDFs

legacy/                           # Archived notebooks and pre-V1 scripts
```

## CLI

```bash
cd recsocial_py
python -m recsocial.cli v1 preprocess --config configs/v1.yaml
python -m recsocial.cli v2 experiment --config configs/v2.yaml
python -m recsocial.cli v3 experiment --config configs/v3.yaml
```

## Documentation

- [V1 plan & notes](docs/v1/V1_REPRODUCTION_PLAN.md)
- [V2 notes](docs/v2/reproduction_notes_v2.md)
- [V3 notes](docs/v3/reproduction_notes_v3.md)
- [Gap analysis](docs/v1/GAP_ANALYSIS.md)

## License

See [LICENSE](LICENSE).
