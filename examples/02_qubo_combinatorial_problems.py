"""Example 2: Quadratic Unconstrained Binary Optimization Formulations.

Demonstrates:
1. Traveling Salesperson Problem (TSP) mapping to QUBO.
2. Graph 3-Coloring Formulation.
3. 0/1 Knapsack problem and Ising conversion.
"""

import numpy as np
import networkx as nx

from pqblocks_qubo import (
    BinaryQuadraticModel,
    Vartype,
    to_qubo,
    to_ising,
    load_tsp,
    load_coloring,
    load_knapsack,
)

def main():
    print("1. Formulating 4-City TSP as QUBO...")
    # Distance matrix for 4 cities
    dist_matrix = np.array([
        [0.0, 10.0, 15.0, 20.0],
        [10.0, 0.0, 35.0, 25.0],
        [15.0, 35.0, 0.0, 30.0],
        [20.0, 25.0, 30.0, 0.0],
    ])
    tsp_bqm, var_map = load_tsp(dist_matrix)
    print(f"   TSP variables: {tsp_bqm.num_variables}, linear terms: {len(tsp_bqm.linear)}, quadratic terms: {len(tsp_bqm.quadratic)}")

    print("\n2. Graph Coloring Formulation...")
    G = nx.petersen_graph()
    coloring_bqm = load_coloring(G, num_colors=3)
    print(f"   Petersen graph 3-coloring variables: {coloring_bqm.num_variables}")
    print(f"   Offset energy: {coloring_bqm.offset:.2f}")

    print("\n3. Knapsack Formulation & Spin/Binary Duality...")
    values = [10, 20, 30]
    weights = [1, 2, 3]
    capacity = 4
    knapsack_bqm = load_knapsack(values, weights, capacity)
    print(f"   Knapsack BQM variables: {knapsack_bqm.num_variables}, offset: {knapsack_bqm.offset:.1f}")

    # Convert to Ising and back
    ising_model = to_ising(knapsack_bqm)
    qubo_recovered = to_qubo(ising_model)
    print(f"   Converted to Ising (vartype={ising_model.vartype.name}) and back (vartype={qubo_recovered.vartype.name})")

    # Evaluate sample energy
    sample = {i: 1 if i in [0, 2] else 0 for i in range(knapsack_bqm.num_variables)}
    print(f"   Sample energy: {knapsack_bqm.energy(sample):.2f}")

    print("\nAll QUBO formulation and conversion pipelines completed successfully.")

if __name__ == "__main__":
    main()
