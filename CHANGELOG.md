# Changelog: pqblocks-qubo

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-26

### Added
- Initial public release of `pqblocks-qubo`.
- Unified `BinaryQuadraticModel` (BQM) supporting both SPIN and BINARY variable types.
- Exact bidirectional converters (`to_qubo`, `to_ising`, `from_ising`, `from_qubo`).
- Linear constraint compilation routines (`compile_equality`, `compile_inequality`, `compile_onehot`).
- Automated penalty factor tuning (`PenaltyAutoTuner`).
- Standard NP-hard problem loaders (MaxCut, Traveling Salesperson, Graph Coloring, Knapsack, Portfolio Optimization).
- 83 unit and integration tests achieving 93% statement coverage.
