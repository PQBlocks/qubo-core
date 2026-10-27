# Security Notes: pqblocks-qubo

## Classification & Domain

> [!IMPORTANT]
> **Not Cryptography**: `pqblocks-qubo` is a classical mathematical modeling and combinatorial optimization compilation library. It does not implement cryptography, generate encryption keys, or secure data.

## Computational Safety

- **Memory Limits**: Large problem matrices (e.g. $N > 10,000$) create dense $O(N^2)$ quadratic arrays. Ensure sufficient host memory when compiling dense formulations.
- **Floating-Point Determinism**: Energy evaluations and conversions use IEEE-754 double-precision arithmetic (`numpy.float64`). Minor numerical precision variations may occur across CPU microarchitectures.
