# Mathematical Evidence: pqblocks-qubo

## Theoretical Foundations

### 1. Duality between Ising and QUBO

The Ising Hamiltonian is defined over spin variables $s_i \in \{-1, +1\}$:
$$H_{\text{Ising}}(s) = \sum_{i < j} J_{ij} s_i s_j + \sum_i h_i s_i + c_{\text{Ising}}$$

The QUBO cost function is defined over binary variables $x_i \in \{0, 1\}$:
$$H_{\text{QUBO}}(x) = \sum_{i \le j} Q_{ij} x_i x_j + c_{\text{QUBO}}$$

Using the linear transformation $s_i = 2x_i - 1$:
$$s_i s_j = (2x_i - 1)(2x_j - 1) = 4 x_i x_j - 2x_i - 2x_j + 1$$

Substituting into the Hamiltonian gives the exact parameter conversions implemented in `pqblocks_qubo.model.converters`:
- $Q_{ij} = 4 J_{ij}$ for $i < j$
- $Q_{ii} = 2 h_i - 2 \sum_{j \neq i} J_{ij}$
- $c_{\text{QUBO}} = c_{\text{Ising}} - \sum_i h_i + \sum_{i < j} J_{ij}$

### 2. Constraint Compilation

- **Equality Constraint** ($\sum_i a_i x_i = b$):
  $$P \cdot \left(\sum_i a_i x_i - b\right)^2 = P \cdot \left(\sum_i a_i^2 x_i + 2 \sum_{i < j} a_i a_j x_i x_j - 2b \sum_i a_i x_i + b^2\right)$$
- **Inequality Constraint** ($\sum_i a_i x_i \le b$):
  Introduces binary slack variables $y_k \in \{0, 1\}$ with powers of two weights $\sum_{k=0}^M 2^k y_k$ converting the inequality into an equality constraint.

## Normative References

1. Lucas, A. (2014). *Ising formulations of many NP problems*. Frontiers in Physics, 2, 5.
2. Glover, F., Kochenberger, G., & Du, Y. (2019). *Quantum Bridge Analytics I: a tutorial on formulating and using QUBO models*. 4OR, 17(4), 335-371.
3. D-Wave Systems. *Binary Quadratic Model Architecture and Reference*.
