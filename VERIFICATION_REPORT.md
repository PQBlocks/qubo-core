# Verification Report: pqblocks-qubo

This report documents verification results across supported Python runtimes.

## Verification Matrix

- **Package**: `pqblocks-qubo`
- **Version**: 0.1.0
- **Domain**: Classical Optimization (QUBO/Ising models)
- **Test Suite**: 83 automated tests
- **Coverage**: 93% statement coverage (862/925 statements)

| Python Version | Wheel Build | Test Suite | Examples Execution | Gate Status |
|---|---|---|---|---|
| Python 3.11 | PASS | 83 passed | 2 examples passed | **PASS** |
| Python 3.12 | PASS | 83 passed | 2 examples passed | **PASS** |
| Python 3.13 | PASS | 83 passed | 2 examples passed | **PASS** |
| Python 3.14 | PASS | 83 passed | 2 examples passed | **PASS** |

## Executed Examples

1. `examples/quickstart.py`: Validates MaxCut graph formulation, Ising-to-QUBO conversion, and state energy computation.
2. `examples/02_qubo_combinatorial_problems.py`: Validates Traveling Salesperson Problem compilation, graph coloring, and 0/1 knapsack duality.

## Zero Leaks Verification

Scanned repository files against internal patterns and markers. Result: **0 leak hits**.
