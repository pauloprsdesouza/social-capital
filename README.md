# Social Capital Recommender Reproduction

Reproduce, validate, and extend three published Social Capital recommender papers from a single CSV-based Python package.

| Version | Paper | Validation |
|---------|-------|------------|
| **V1** | [FedCSIS 2022](docs/papers/Exploiting%20Social%20Capital%20for%20Recommendation%20in%20Social%20Networks.pdf) | 12 trial metrics |
| **V2** | [AMCIS 2024](docs/papers/Unlocking%20the%20Power%20of%20Social%20Capital%20Advanced%20Strategies%20for%20Enhanced%20Personalized%20Recommendations%20in%20Online%20Social%20Networks.pdf) | 96 figure metrics (Figs 3–10) |
| **V3** | [SCSA-PLUS + PCA](docs/papers/Exploiting%20Social%20Capital%20for%20Improving%20Personalized%20Recommendations%20in%20Online%20Social%20Networks.pdf) | 7 headline metrics |

## Quick start

```bash
cd recsocial_py
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e ".[dev]"

python -m recsocial.cli run all    # pipeline + validation
```

Reports are written to `recsocial_py/reports/` (generated locally, not committed).

## Documentation

See the [documentation index](docs/README.md).

| Guide | Description |
|-------|-------------|
| [Getting started](docs/GETTING_STARTED.md) | Install, run, validate |
| [Tutorial](docs/TUTORIAL.md) | Data layer and custom algorithms |
| [Version comparison](docs/VERSION_COMPARISON.md) | V1 vs V2 vs V3 |
| [Evaluation](docs/EVALUATION.md) | Metrics, protocols, paper targets |
| [Architecture](docs/ARCHITECTURE.md) | Code layout |
| [Improving algorithms](docs/IMPROVING.md) | Tune and extend |

## Repository layout

```text
recsocial_py/          Python package
  configs/             YAML settings
  data/raw/            Source CSVs
  src/recsocial/       shared/ + slices/v1|v2|v3/
docs/                  Guides and paper SDDs
```

## License

See [LICENSE](LICENSE).
