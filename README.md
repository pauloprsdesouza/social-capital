# Social Capital Recommender Reproduction

Reproduce, validate, and extend three published Social Capital recommender papers — from a single CSV-based Python package.

| Version | Paper | Status |
|---------|-------|--------|
| **V1** | [FedCSIS 2022](docs/papers/Exploiting%20Social%20Capital%20for%20Recommendation%20in%20Social%20Networks.pdf) | Passes all paper targets |
| **V2** | [AMCIS 2024](docs/papers/Unlocking%20the%20Power%20of%20Social%20Capital%20Advanced%20Strategies%20for%20Enhanced%20Personalized%20Recommendations%20in%20Online%20Social%20Networks.pdf) | Partial — trends match, some numeric gaps |
| **V3** | [SCSA-PLUS + PCA](docs/papers/Exploiting%20Social%20Capital%20for%20Improving%20Personalized%20Recommendations%20in%20Online%20Social%20Networks.pdf) | Partial — headline metrics match |

## Quick start

```bash
cd recsocial_py
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e ".[dev]"

# Run all three versions + validate against papers
python -m recsocial.cli run all
```

Reports are generated under `recsocial_py/reports/` (not committed — regenerate anytime).

## Documentation

**Start with the [documentation index](docs/README.md).**

| Guide | Description |
|-------|-------------|
| [Getting started](docs/GETTING_STARTED.md) | Install, run, validate |
| [Tutorial: data & algorithms](docs/TUTORIAL.md) | CSV data layer, run all versions, custom algorithms on V3 |
| [Version comparison](docs/VERSION_COMPARISON.md) | V1 vs V2 vs V3 |
| [Architecture](docs/ARCHITECTURE.md) | Code layout and data flow |
| [Evaluation](docs/EVALUATION.md) | Metrics and paper targets |
| [Improving algorithms](docs/IMPROVING.md) | Tune and extend |

## Repository layout

```text
recsocial_py/          Python package — all active code
  configs/             YAML settings (v1, v2, v3)
  data/raw/            Source CSVs (ratings, tweets, users)
  src/recsocial/       shared/ + slices/v1|v2|v3/
  tests/
docs/                  Guides and paper SDDs
```

## License

See [LICENSE](LICENSE).
