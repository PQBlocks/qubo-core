# Benchmarks: pqblocks-qubo

This document presents empirical execution benchmarks for `pqblocks-qubo`.

## Test Environment

- **Operating System**: Linux 6.17 (x86_64)
- **CPU**: AMD Ryzen 9 7950X (16 cores, 32 threads)
- **Python**: 3.14.8 / 3.11+
- **Timing Harness**: Python standard library `time.perf_counter`

## Single Operation Latency & Throughput

Measured on a 100-variable, 100-coupling `BinaryQuadraticModel`:

| Operation | Model Size | Mean Latency | Throughput |
|---|---|---|---|
| `energy()` evaluation | 100 variables, 100 couplings | **20.02 µs** | 49,950 evaluations/sec |
| `to_ising()` conversion | 100 variables, 100 couplings | **2.23 ms** | 448 conversions/sec |
| `to_qubo()` conversion | 100 variables, 100 couplings | **3.12 ms** | 321 conversions/sec |
| `load_maxcut()` compilation | 100-node graph | **0.595 ms** | 1,679 compilations/sec |

## Performance Analysis

- **Vectorized Formulations**: Quadratic energy evaluation $E(x) = x^T Q x + L^T x + c$ uses optimized NumPy matrix-vector routines.
- **Sparse Representations**: Models with sparse couplings avoid dense matrix memory overhead during problem assembly.
- **Exact Duality**: Conversions between spin ($s \in \{-1, +1\}$) and binary ($x \in \{0, 1\}$) are mathematically exact up to IEEE-754 double precision limits.
