# Documentation index

This folder contains everything needed to **understand**, **reproduce**, **validate**, and **extend** the Social Capital recommender across three paper versions.

## Start here

| Document | Audience | Purpose |
|----------|----------|---------|
| [Getting started](GETTING_STARTED.md) | Everyone | Install, run, validate, read reports |
| [Tutorial: data & algorithms](TUTORIAL.md) | Practitioners | Run all versions, use V3 CSV data, build custom algorithms |
| [Version comparison](VERSION_COMPARISON.md) | Researchers | What changed from V1 → V2 → V3 |
| [Architecture](ARCHITECTURE.md) | Developers | Code layout, data flow, module map |
| [Evaluation](EVALUATION.md) | Validators | Metrics, protocols, paper targets |
| [Improving algorithms](IMPROVING.md) | Contributors | Tune weights, add variants, test changes |

## Papers and SDDs

| Version | Paper | SDD / notes |
|---------|-------|-------------|
| **V1** | [FedCSIS 2022 baseline](papers/Exploiting%20Social%20Capital%20for%20Recommendation%20in%20Social%20Networks.pdf) | [SDD markdown](papers/SDD%20-%20Exploiting%20Social%20Capital%20for%20Recommendation%20in%20Social%20Networks.md) · [V1 notes](v1/reproduction_notes.md) |
| **V2** | [AMCIS 2024 enhanced SC](papers/Unlocking%20the%20Power%20of%20Social%20Capital%20Advanced%20Strategies%20for%20Enhanced%20Personalized%20Recommendations%20in%20Online%20Social%20Networks.pdf) | [SDD markdown](papers/Unlocking%20the%20Power%20of%20Social%20Capital%20Advanced%20Strategies%20for%20Enhanced%20Personalized%20Recommendations%20in%20Online%20Social%20Networks.md) · [V2 notes](v2/reproduction_notes_v2.md) |
| **V3** | [SCSA-PLUS + PCA](papers/Exploiting%20Social%20Capital%20for%20Improving%20Personalized%20Recommendations%20in%20Online%20Social%20Networks.pdf) | [SDD markdown](papers/Exploiting%20Social%20Capital%20for%20Improving%20Personalized%20Recommendations%20in%20Online%20Social%20Networks.md) · [V3 notes](v3/reproduction_notes_v3.md) |

## Package entry point

All active code lives in [`recsocial_py/`](../recsocial_py/). See [`recsocial_py/README.md`](../recsocial_py/README.md) for CLI commands.
