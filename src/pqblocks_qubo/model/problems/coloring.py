"""Problem loader for Graph Coloring problems."""

import numpy as np
import networkx as nx
from typing import Union, Dict, Tuple, List, Optional
from ..bqm import BinaryQuadraticModel
from ..vartype import Vartype
from ..constraints import compile_onehot
from ..auto_tuner import PenaltyAutoTuner


def load_coloring(
    graph: Union[nx.Graph, np.ndarray, List[Tuple[int, int]]], 
    num_colors: int, 
    A: Optional[float] = None
) -> BinaryQuadraticModel:
    """Load a Graph Coloring problem as a BinaryQuadraticModel.
    
    Creates a QUBO formulation where exactly one color is assigned to each vertex,
    and adjacent vertices cannot share the same color.
    
    Args:
        graph: NetworkX Graph, adjacency matrix, or list of edges.
        num_colors: Number of colors to use.
        A: Penalty weight for constraints. If None, auto-tuned.
        
    Returns:
        BinaryQuadraticModel representing the coloring problem.
    """
    # Normalize graph to NetworkX Graph
    if isinstance(graph, np.ndarray):
        nx_graph = nx.from_numpy_array(graph)
    elif isinstance(graph, list):
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(graph)
    else:
        nx_graph = graph.copy()
    
    n_vertices = nx_graph.number_of_nodes()
    
    # Create variable mapping: vertex * num_colors + color
    def var_id(vertex: int, color: int) -> int:
        return vertex * num_colors + color
    
    # Build one-hot constraints for each vertex (exactly one color per vertex)
    constraints = []
    for v in range(n_vertices):
        color_vars = [var_id(v, c) for c in range(num_colors)]
        if A is not None:
            constraints.append(compile_onehot(color_vars, A))
        # If A is None, we'll handle auto-tuning later
    
    # Build adjacency penalty terms (adjacent vertices cannot share color)
    quadratic = {}
    for u, v in nx_graph.edges():
        for c in range(num_colors):
            var_u = var_id(u, c)
            var_v = var_id(v, c)
            # Penalty weight for adjacent vertices sharing color
            # If A is provided, use it; otherwise we'll determine it during auto-tuning
            penalty_weight = A if A is not None else 1.0
            quadratic[(var_u, var_v)] = penalty_weight
    
    # If A is provided, we can directly construct the BQM
    if A is not None:
        # Combine all constraint BQMs
        combined_constraints = constraints[0]
        for constraint in constraints[1:]:
            # Manual combination since we don't have __add__ implemented yet
            combined_constraints = _combine_bqms(combined_constraints, constraint)
        
        # Create BQM for adjacency penalties
        adjacency_bqm = BinaryQuadraticModel({}, quadratic, 0.0, Vartype.BINARY)
        
        # Combine constraints and adjacency penalties
        final_bqm = _combine_bqms(combined_constraints, adjacency_bqm)
        return final_bqm
    
    # If A is None, we need to auto-tune
    # For auto-tuning, we need to create an objective function and constraint function
    # The objective is just the adjacency penalties
    objective = BinaryQuadraticModel({}, quadratic, 0.0, Vartype.BINARY)
    
    # Define constraint function for auto-tuner
    def build_constraints(A_val: float) -> BinaryQuadraticModel:
        vertex_constraints = []
        for v in range(n_vertices):
            color_vars = [var_id(v, c) for c in range(num_colors)]
            vertex_constraints.append(compile_onehot(color_vars, A_val))
        
        # Combine all vertex constraints
        combined = vertex_constraints[0]
        for constraint in vertex_constraints[1:]:
            combined = _combine_bqms(combined, constraint)
        return combined
    
    # Define feasibility check
    def is_feasible(sample: Dict[int, int]) -> bool:
        # Check one-hot constraint for each vertex
        for v in range(n_vertices):
            color_sum = sum(sample.get(var_id(v, c), 0) for c in range(num_colors))
            if color_sum != 1:
                return False
        return True
    
    # Run auto-tuner
    tuner = PenaltyAutoTuner(
        objective_bqm=objective,
        constraint_bqm_func=build_constraints,
        feasibility_check=is_feasible,
        max_iters=20
    )
    tuned_A, _, _ = tuner.run()
    
    # Rebuild with tuned A
    return load_coloring(graph, num_colors, tuned_A)


def _combine_bqms(bqm1: BinaryQuadraticModel, bqm2: BinaryQuadraticModel) -> BinaryQuadraticModel:
    """Combine two BQMs using key-based lookup combination.
    
    This implementation correctly handles non-contiguous, negative, and slack variables
    by using variable IDs and variable_index mappings, similar to the correct implementation in tsp.py.
    
    Args:
        bqm1: First BinaryQuadraticModel.
        bqm2: Second BinaryQuadraticModel.
        
    Returns:
        Combined BinaryQuadraticModel.
        
    Raises:
        ValueError: If the BQMs have different variable types.
    """
    if bqm1.vartype != bqm2.vartype:
        raise ValueError("Cannot combine BQMs with different variable types")
        
    # Create a new combined structure
    combined_variable_ids = list(bqm1.variable_ids)  # Make a copy
    combined_variable_index = dict(bqm1.variable_index)  # Make a copy
    
    # Add bqm2's variables to the combined structure
    for var_id in bqm2.variable_ids:
        if var_id not in combined_variable_index:
            combined_variable_index[var_id] = len(combined_variable_ids)
            combined_variable_ids.append(var_id)
            
    # Create combined linear array
    n_combined = len(combined_variable_ids)
    linear = np.zeros(n_combined, dtype=np.float64)
    
    # Copy bqm1 linear terms
    for i, var_id in enumerate(bqm1.variable_ids):
        val = bqm1.linear[i]
        idx = combined_variable_index[var_id]
        linear[idx] += val
            
    # Add bqm2 linear terms
    for i, var_id in enumerate(bqm2.variable_ids):
        val = bqm2.linear[i]
        idx = combined_variable_index[var_id]
        linear[idx] += val
            
    # Combine quadratic terms
    # Create combined quadratic matrix
    quadratic = np.zeros((n_combined, n_combined), dtype=np.float64)
    
    # Copy bqm1 quadratic terms
    n1 = bqm1.num_variables
    for i in range(n1):
        for j in range(n1):
            if bqm1.quadratic[i, j] != 0:
                var_i = bqm1.variable_ids[i]
                var_j = bqm1.variable_ids[j]
                idx_i = combined_variable_index[var_i]
                idx_j = combined_variable_index[var_j]
                quadratic[idx_i, idx_j] += bqm1.quadratic[i, j]
    
    # Add bqm2 quadratic terms
    n2 = bqm2.num_variables
    for i in range(n2):
        for j in range(n2):
            if bqm2.quadratic[i, j] != 0:
                var_i = bqm2.variable_ids[i]
                var_j = bqm2.variable_ids[j]
                idx_i = combined_variable_index[var_i]
                idx_j = combined_variable_index[var_j]
                quadratic[idx_i, idx_j] += bqm2.quadratic[i, j]
                
    # Combine offsets
    offset = bqm1.offset + bqm2.offset
    
    return BinaryQuadraticModel(linear, quadratic, offset, bqm1.vartype)