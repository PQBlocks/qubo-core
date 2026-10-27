# Use Cases: pqblocks-qubo

`pqblocks-qubo` serves as the foundational modeling layer for quantum-inspired heuristics (Simulated Annealing, Simulated Bifurcation, Digital Annealing).

## 1. Network MaxCut Partitioning

Compile graph partitioning problems into QUBO matrices for solver consumption:

```python
import networkx as nx
from pqblocks_qubo import load_maxcut, to_qubo

# Construct communication or cluster network
G = nx.erdos_renyi_graph(50, 0.15)
maxcut_bqm = load_maxcut(G)
qubo_model = to_qubo(maxcut_bqm)

# Access matrix elements for solvers
linear_weights = qubo_model.linear
quadratic_couplings = qubo_model.quadratic
```

## 2. Supply Chain Routing (TSP)

Convert travel distance matrices into binary permutation constraint models:

```python
import numpy as np
from pqblocks_qubo import load_tsp

distances = np.random.uniform(5.0, 50.0, size=(10, 10))
np.fill_diagonal(distances, 0.0)

tsp_model, mapping = load_tsp(distances)
```

## 3. Financial Portfolio Selection

Formulate mean-variance portfolio optimization problems with cardinality and budget constraints:

```python
import numpy as np
from pqblocks_qubo.model.problems.portfolio import load_portfolio

expected_returns = np.array([0.12, 0.18, 0.10, 0.15, 0.08])
cov_matrix = np.eye(5) * 0.05
budget = 3

portfolio_bqm = load_portfolio(expected_returns, cov_matrix, budget=budget, risk_factor=0.5)
```
