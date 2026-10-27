# Testing: pqblocks-qubo

## Test Harness Overview

`pqblocks-qubo` includes an automated suite of 83 unit and integration tests verifying model manipulation, conversion identities, constraint compilers, and problem loaders.

### Test Execution

Run the suite with pytest:

```bash
pytest -v
```

With coverage measurement:

```bash
pytest --cov=pqblocks_qubo --cov-report=term-missing
```

## Coverage Report

The test suite achieves **93% statement coverage** across all production source files:

| Module | Statements | Missing | Coverage |
|---|---|---|---|
| `pqblocks_qubo/__init__.py` | 11 | 0 | **100%** |
| `pqblocks_qubo/model/__init__.py` | 5 | 0 | **100%** |
| `pqblocks_qubo/model/auto_tuner.py` | 110 | 2 | **98%** |
| `pqblocks_qubo/model/bqm.py` | 110 | 16 | **85%** |
| `pqblocks_qubo/model/constraints.py` | 110 | 7 | **94%** |
| `pqblocks_qubo/model/converters.py` | 109 | 17 | **84%** |
| `pqblocks_qubo/model/vartype.py` | 6 | 0 | **100%** |
| `pqblocks_qubo/model/problems/coloring.py` | 95 | 2 | **98%** |
| `pqblocks_qubo/model/problems/knapsack.py` | 99 | 6 | **94%** |
| `pqblocks_qubo/model/problems/maxcut.py` | 70 | 9 | **87%** |
| `pqblocks_qubo/model/problems/portfolio.py` | 93 | 3 | **97%** |
| `pqblocks_qubo/model/problems/tsp.py` | 107 | 1 | **99%** |
| **Total** | **925** | **63** | **93%** |

## Test Scenarios Covered

1. **Model Arithmetic & Equivalence**:
   - In-place and out-of-place BQM addition, scaling, and offset updates.
   - Exact identity verification across `to_qubo(to_ising(bqm)) == bqm`.
2. **Constraint Enforcement**:
   - Linear equality constraints ($\sum c_i x_i = k$).
   - Inequality constraints with slack variable expansions ($\sum c_i x_i \le k$).
   - One-hot constraints ($\sum x_i = 1$).
3. **Problem Formulation Correctness**:
   - MaxCut partition ground states on cycle and complete graphs.
   - TSP permutation matrices and sub-tour penalty verification.
   - Graph Coloring non-conflict constraints.
   - 0/1 Knapsack value-to-weight optimization.
   - Markowitz portfolio variance/return trade-off formulations.
4. **Golden Instance Verification**:
   - Validated against pre-computed golden solution states.
