"""TSP problem loader."""

import numpy as np
from typing import Optional, Dict, Tuple, Callable, Any
from ..bqm import BinaryQuadraticModel
from ..vartype import Vartype
from ..constraints import compile_onehot


def _combine_bqms(bqm1: BinaryQuadraticModel, bqm2: BinaryQuadraticModel) -> BinaryQuadraticModel:
    """Combine two BinaryQuadraticModels by adding their terms.
    
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
        
    # Create a new combined structure, preserving order
    combined_variable_ids = list(bqm1.variable_ids)  # Make a copy of bqm1's variables
    
    # Add bqm2's variables in order, but don't sort!
    for var_id in bqm2.variable_ids:
        if var_id not in bqm1.variable_index:  # Check against bqm1's index to avoid duplicates
            combined_variable_ids.append(var_id)
    
    # Create variable_index mapping that preserves order
    combined_variable_index = {var_id: idx for idx, var_id in enumerate(combined_variable_ids)}
            
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
    
    # Create the combined BQM manually to avoid the sorting in the constructor
    combined_bqm = BinaryQuadraticModel.__new__(BinaryQuadraticModel)
    combined_bqm.linear = linear
    combined_bqm.quadratic = quadratic
    combined_bqm.offset = offset
    combined_bqm.vartype = bqm1.vartype
    combined_bqm.num_variables = n_combined
    combined_bqm.variable_ids = combined_variable_ids
    combined_bqm.variable_index = combined_variable_index
    
    return combined_bqm


def load_tsp(distance_matrix: np.ndarray, A: Optional[float] = None, B: float = 1.0) -> Tuple[BinaryQuadraticModel, Dict[Tuple[int, int], int]]:
    """Load a Traveling Salesperson Problem into a BinaryQuadraticModel.
    
    Creates an N×N permutation matrix constraint model where:
    - x_{v,j} = 1 if city v is visited at time step j
    - Each city is visited exactly once (row one-hot constraint)
    - Each time step has exactly one city (column one-hot constraint)
    
    The objective function is:
    H = B * Σ_{u,v,j} d_{u,v} * x_{u,j} * x_{v,j+1} + A * (constraints)
    
    Args:
        distance_matrix: N×N matrix of distances between cities.
        A: Penalty weight for constraints. If None, uses 2.0 as default.
        B: Weight for objective function (default 1.0).
        
    Returns:
        Tuple of (BinaryQuadraticModel representing the TSP, variable mapping dict).
        
    Raises:
        ValueError: If distance_matrix is not square or A <= 0.
    """
    # Validate inputs
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be 2D and square")
        
    if A is not None and A <= 0:
        raise ValueError("Penalty weight A must be positive")
        
    if B <= 0:
        raise ValueError("Objective weight B must be positive")
        
    N = distance_matrix.shape[0]
    
    # Handle edge cases
    if N == 0:
        return BinaryQuadraticModel({}, {}, 0.0, Vartype.BINARY), {}
        
    if N == 1:
        # Trivial case: one city
        var_map = {(0, 0): 0}
        linear = {0: 0.0}
        quadratic = {}
        offset = 0.0
        bqm = BinaryQuadraticModel(linear, quadratic, offset, Vartype.BINARY)
        return bqm, var_map
    
    # Convert distance matrix to integers if it's not already
    # This is important for slack encoding in constraints
    distance_matrix = np.rint(distance_matrix).astype(np.int64)
    
    # Create permutation matrix variables: x_{v,j}
    var_map = {}
    var_id = 0
    for v in range(N):
        for j in range(N):
            var_map[(v, j)] = var_id
            var_id += 1
    
    # Build objective function
    # Initialize objective_linear with all variables to ensure correct ordering
    objective_linear = {i: 0.0 for i in range(N * N)}
    objective_quadratic = {}
    
    # Distance terms: B * Σ_{u,v,j} d_{u,v} * x_{u,j} * x_{v,j+1}
    for u in range(N):
        for v in range(N):
            if u == v:
                continue
            dist = distance_matrix[u, v]
            for j in range(N):
                u_j = var_map[(u, j)]
                v_j1 = var_map[(v, (j + 1) % N)]  # Wrap around: N -> 0
                objective_quadratic[(u_j, v_j1)] = B * dist
    
    objective = BinaryQuadraticModel(objective_linear, objective_quadratic, 0.0, Vartype.BINARY)
    
    # Build constraint functions for auto-tuning or fixed penalty
    def build_constraints(A_val: float) -> BinaryQuadraticModel:
        """Build all constraints with given penalty weight."""
        # City constraints: each city visited exactly once
        city_constraints = []
        for v in range(N):
            vars_v = [var_map[(v, j)] for j in range(N)]
            constraint_bqm = compile_onehot(vars_v, A_val)
            city_constraints.append(constraint_bqm)
        
        # Time constraints: each time step has exactly one city
        time_constraints = []
        for j in range(N):
            vars_j = [var_map[(v, j)] for v in range(N)]
            constraint_bqm = compile_onehot(vars_j, A_val)
            time_constraints.append(constraint_bqm)
        
        # Combine all constraints
        combined = city_constraints[0]
        for c in city_constraints[1:] + time_constraints:
            combined = _combine_bqms(combined, c)
        return combined
    
    # Apply constraints with either provided A or default
    if A is None:
        # Use default penalty weight
        A = 2.0
    
    combined_constraints = build_constraints(A)
    
    return _combine_bqms(objective, combined_constraints), var_map