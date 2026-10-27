# Agents Guide: pqblocks-qubo

This guide provides automated test and integration instructions for `pqblocks-qubo`.

## Validation Commands

```bash
# Run test suite
pytest -v

# Check coverage
pytest --cov=pqblocks_qubo --cov-fail-under=90

# Run examples
python examples/quickstart.py
python examples/02_qubo_combinatorial_problems.py
```

## Public API Symbols

Exports from `pqblocks_qubo`:
- `BinaryQuadraticModel`, `Vartype`
- `to_qubo`, `to_ising`, `from_ising`, `from_qubo`
- `compile_equality`, `compile_inequality`, `compile_onehot`
- `PenaltyAutoTuner`
- `load_maxcut`, `load_tsp`, `load_coloring`, `load_knapsack`, `load_portfolio`
