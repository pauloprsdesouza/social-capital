# recsocial

Python package reproducing three Social Capital recommender papers (V1 / V2 / V3).

Full documentation: **[../docs/README.md](../docs/README.md)** · Tutorial: **[../docs/TUTORIAL.md](../docs/TUTORIAL.md)**

## Install

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

## Run

```bash
# All versions + paper validation
python -m recsocial.cli run all

# Single version
python -m recsocial.cli run v1
python -m recsocial.cli run v2
python -m recsocial.cli run v3

# Validate only
python -m recsocial.cli validate
```

## Outputs

| Path | Content |
|------|---------|
| `reports/v1/report.md` | FedCSIS 2022 results |
| `reports/v2/report.md` | AMCIS 2024 results + figure validation |
| `reports/v3/report.md` | SCSA-PLUS + PCA results |
| `reports/validation_summary.md` | Cross-paper validation |
| `reports/v*/figures/` | Publication-style charts |

## Package structure

```text
src/recsocial/
  cli.py                 Entry point (recsocial command)
  shared/                Evaluation, pipeline, validation, visualization
    algorithms.py        Canonical algorithm names
    pipeline.py          run_v1|v2|v3_pipeline, run_all_pipelines
    paper_validation.py  Cross-paper target comparison
  slices/
    v1/                  FedCSIS baseline (migrate, social_capital, experiment)
    v2/                  AMCIS enhanced components + re-ranking
    v3/                  SCSA-PLUS + PCA + statistics
configs/
  v1.yaml, v2.yaml, v3.yaml
  reference_results.yaml   V2 AMCIS figure targets (Figs 3–10)
```

## Config files

| File | Version | Key settings |
|------|---------|--------------|
| `configs/v1.yaml` | V1 | influence, sentiment, `paper_targets`, `map_protocol: fedcsis_pooled` |
| `configs/v2.yaml` | V2 | component weights, STATE_ART, `reference_results.yaml` validation |
| `configs/v3.yaml` | V3 | SCSA-PLUS, PCA, `paper_targets`, paired t-tests |

## CLI reference

| Command | Description |
|---------|-------------|
| `recsocial run all` | Full pipeline V1→V2→V3 + validation |
| `recsocial run v1\|v2\|v3` | Single slice pipeline |
| `recsocial validate` | Compare all reports to paper targets |
| `recsocial v1 preprocess\|score\|experiment\|plot\|report` | V1 granular steps |
| `recsocial v2 preprocess\|score\|experiment\|plot` | V2 granular steps |
| `recsocial v3 preprocess\|score\|experiment\|plot` | V3 granular steps |

## Python API

```python
from recsocial.shared.pipeline import run_all_pipelines, package_root
from recsocial.shared.paper_validation import validate_all_papers
from recsocial.slices.v2.experiment import run_v2_experiment
from recsocial.slices.v2.config import load_v2_config

root = package_root()
results = validate_all_papers(root)
```

## Further reading

- [Version comparison](../docs/VERSION_COMPARISON.md) — differences among V1, V2, V3
- [Architecture](../docs/ARCHITECTURE.md) — module map and data flow
- [Improving algorithms](../docs/IMPROVING.md) — how to tune and extend
