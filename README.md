# pqblocks-qubo

> Unified mathematical representation and converter library for Quadratic Unconstrained Binary Optimization (QUBO) and Ising formulations.

[![CI](https://github.com/PQBlocks/qubo-core/actions/workflows/ci.yml/badge.svg)](https://github.com/PQBlocks/qubo-core/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## Overview

`pqblocks-qubo` is a foundational mathematical model and compilation engine for Quadratic Unconstrained Binary Optimization (QUBO) and Ising spin-glass problems.

It provides a unified `BinaryQuadraticModel` (BQM) abstraction supporting both binary variables ($x_i \in \{0, 1\}$) and spin variables ($s_i \in \{-1, +1\}$), with seamless $O(1)$ bidirectional conversions, linear constraint compilation (equality, inequality, one-hot), and canonical problem generators (MaxCut, TSP, Graph Coloring, 0/1 Knapsack, Portfolio Optimization).

### Key Highlights

- **BQM Abstraction**: Represents quadratic energy functions with linear terms, symmetric quadratic couplings, and constant energy offsets.
- **Spin/Binary Duality**: Instant mathematical conversion between Ising and QUBO representations without loss of precision.
- **Constraint Compilers**: Automatic translation of linear equality and inequality constraints into quadratic penalty terms.
- **Problem Formulations**: Built-in loaders for NP-hard combinatorial optimization challenges.

---

## Installation

```bash
pip install pqblocks-qubo
```

### Dependencies

Requires Python 3.11+, `numpy>=1.20.0`, and `networkx`.

---

## Quickstart

```python
import networkx as nx
from pqblocks_qubo import load_maxcut, to_qubo

# 1. Build a graph and construct native Ising model for MaxCut
G = nx.cycle_graph(4)
bqm_ising = load_maxcut(G)

# 2. Convert transparently to QUBO
bqm_qubo = to_qubo(bqm_ising)

# 3. Evaluate state energy (deterministic evaluation)
sample = {0: 0, 1: 1, 2: 0, 3: 1}
print(f"QUBO Energy: {bqm_qubo.energy(sample)}")
```

---

## Combinatorial Problem Modeling

### Traveling Salesperson Problem (TSP)

```python
import numpy as np
from pqblocks_qubo import load_tsp

dist_matrix = np.array([
    [0.0, 10.0, 15.0, 20.0],
    [10.0, 0.0, 35.0, 25.0],
    [15.0, 35.0, 0.0, 30.0],
    [20.0, 25.0, 30.0, 0.0],
])

tsp_bqm, var_map = load_tsp(dist_matrix)
print(f"TSP BQM variables: {tsp_bqm.num_variables}")
```

### 0/1 Knapsack Problem

```python
from pqblocks_qubo import load_knapsack

values = [10, 20, 30]
weights = [1, 2, 3]
capacity = 4

knapsack_bqm = load_knapsack(values, weights, capacity)
print(f"Knapsack BQM variables: {knapsack_bqm.num_variables}")
```

---

## Documentation

- [BENCHMARKS.md](BENCHMARKS.md) - Energy evaluation and conversion microbenchmarks.
- [TESTING.md](TESTING.md) - Test harness details and 93% statement coverage.
- [EVIDENCE.md](EVIDENCE.md) - Theoretical formulations, Hamiltonian definitions, and literature citations.
- [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) - Multi-python validation log.
- [USE_CASES.md](USE_CASES.md) - Practical combinatorial modeling workflows.
- [SECURITY_NOTES.md](SECURITY_NOTES.md) - Scope classification and safe evaluation.
- [AGENTS.md](AGENTS.md) - Automated verification instructions.
- [CHANGELOG.md](CHANGELOG.md) - Release history.

---

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.
