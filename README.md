# Social Capital Recommender Reproduction

Reproduce, validate, and extend three published Social Capital recommender papers from a single CSV-based Python package.

| Version | Paper | Validation |
|---------|-------|------------|
| **V1** | [FedCSIS 2022](docs/papers/Exploiting%20Social%20Capital%20for%20Recommendation%20in%20Social%20Networks.pdf) | 12 trial metrics |
| **V2** | [AMCIS 2024](docs/papers/Unlocking%20the%20Power%20of%20Social%20Capital%20Advanced%20Strategies%20for%20Enhanced%20Personalized%20Recommendations%20in%20Online%20Social%20Networks.pdf) | 96 figure metrics |
| **V3** | [SCSA-PLUS + PCA](docs/papers/Exploiting%20Social%20Capital%20for%20Improving%20Personalized%20Recommendations%20in%20Online%20Social%20Networks.pdf) | 7 headline metrics |

## Quick start

```bash
cd recsocial_py
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e ".[dev]"

python -m recsocial.cli run all
```

Reports are written to `recsocial_py/reports/` (generated locally, not committed).

## Documentation

| Document | Content |
|----------|---------|
| [Guide](docs/GUIDE.md) | Install, run, data layer, custom algorithms, tuning |
| [Versions](docs/VERSIONS.md) | V1 / V2 / V3 comparison and implementation notes |
| [Evaluation](docs/EVALUATION.md) | Metrics, protocols, validation |
| [Architecture](docs/ARCHITECTURE.md) | Code layout and data flow |

Paper SDDs: `docs/papers/`

## Repository layout

```text
recsocial_py/          Python package (configs, data/raw, src, tests)
docs/                  Guides and paper SDDs
```

## License

See [LICENSE](LICENSE).
