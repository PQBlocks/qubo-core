import networkx as nx
from pqblocks_qubo import load_maxcut, to_qubo

# Build graph and load into native Ising model
G = nx.cycle_graph(4)
bqm_ising = load_maxcut(G)

# Convert transparently to QUBO
bqm_qubo = to_qubo(bqm_ising)

# Evaluate state energy (deterministic evaluation)
sample = {0: 0, 1: 1, 2: 0, 3: 1}
print(f"Energy: {bqm_qubo.energy(sample)}")
