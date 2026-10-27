"""Problem loader for Knapsack problems."""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from ..bqm import BinaryQuadraticModel
from ..vartype import Vartype
from ..constraints import compile_equality
from ..auto_tuner import PenaltyAutoTuner, SplitMix64


def load_knapsack(
    values: List[float], 
    weights: List[float], 
    capacity: float, 
    A: Optional[float] = None, 
    B: float = 1.0
) -> BinaryQuadraticModel:
    """Load a Knapsack problem as a BinaryQuadraticModel.
    
    Creates a QUBO formulation for the 0/1 knapsack problem with logarithmic slack encoding.
    
    Args:
        values: List of item values.
        weights: List of item weights.
        capacity: Maximum weight capacity of the knapsack.
        A: Penalty weight for capacity constraint. If None, auto-tuned.
        B: Scaling factor for objective function (default 1.0).
        
    Returns:
        BinaryQuadraticModel representing the knapsack problem.
    """
    n = len(values)
    
    if len(weights) != n:
        raise ValueError("Values and weights lists must have the same length")
    
    if capacity < 0:
        raise ValueError("Capacity must be non-negative")
    
    # Item selection variables
    item_vars = list(range(n))
    
    # Slack variables for capacity: logarithmic encoding
    M = int(np.floor(np.log2(capacity))) if capacity > 0 else 0
    slack_vars = [_get_unique_ancilla_id() for _ in range(M + 1)]
    
    # If A is provided, we can directly construct the BQM
    if A is not None:
        # Weight constraint: Σ weights_i * x_i + Σ coeff_k * y_k = capacity
        capacity_linear = {}
        capacity_quadratic = {}
        
        # Items contribute weight
        for i, w in enumerate(weights):
            capacity_linear[item_vars[i]] = w
        
        # Slack variables encode remaining capacity
        coeffs = [2**m for m in range(M)]
        if M >= 0 and capacity > 0:
            coeffs.append(capacity + 1 - 2**M)
        for var, coeff in zip(slack_vars, coeffs):
            capacity_linear[var] = -coeff
        
        # Build equality constraint: Σ weights_i x_i + Σ coeff_k y_k = capacity
        constraint = compile_equality(list(item_vars) + slack_vars, k=int(capacity), A=A)
        
        # Objective: maximize Σ values_i * x_i → minimize -B * Σ values_i * x_i
        objective_linear = {item_vars[i]: -B * values[i] for i in range(n)}
        objective = BinaryQuadraticModel(objective_linear, {}, 0.0, Vartype.BINARY)
        
        # Combine objective and constraint
        return _combine_bqms(objective, constraint)
    
    # If A is None, we need to auto-tune
    # For auto-tuning, we need to create an objective function and constraint function
    # The objective is to maximize value
    objective_linear = {item_vars[i]: -B * values[i] for i in range(n)}
    objective = BinaryQuadraticModel(objective_linear, {}, 0.0, Vartype.BINARY)
    
    # Define constraint function for auto-tuner
    def build_constraints(A_val: float) -> BinaryQuadraticModel:
        # Weight constraint: Σ weights_i * x_i + Σ coeff_k * y_k = capacity
        capacity_linear = {}
        capacity_quadratic = {}
        
        # Items contribute weight
        for i, w in enumerate(weights):
            capacity_linear[item_vars[i]] = w
        
        # Slack variables encode remaining capacity
        coeffs = [2**m for m in range(M)]
        if M >= 0 and capacity > 0:
            coeffs.append(capacity + 1 - 2**M)
        for var, coeff in zip(slack_vars, coeffs):
            capacity_linear[var] = -coeff
        
        # Build equality constraint: Σ weights_i x_i + Σ coeff_k y_k = capacity
        return compile_equality(list(item_vars) + slack_vars, k=int(capacity), A=A_val)
    
    # Define feasibility check
    def is_feasible(sample: Dict[int, int]) -> bool:
        # Check capacity constraint
        total_weight = sum(weights[i] * sample.get(item_vars[i], 0) for i in range(n))
        return total_weight <= capacity
    
    # Run auto-tuner
    tuner = PenaltyAutoTuner(
        objective_bqm=objective,
        constraint_bqm_func=build_constraints,
        feasibility_check=is_feasible,
        max_iters=20
    )
    tuned_A, _, _ = tuner.run()
    
    # Rebuild with tuned A
    return load_knapsack(values, weights, capacity, tuned_A, B)


# Global counter for generating unique ancilla IDs
_ancilla_counter = 0


def _get_unique_ancilla_id() -> int:
    """Generate a unique negative ancilla ID."""
    global _ancilla_counter
    _ancilla_counter += 1
    return -_ancilla_counter


def _combine_bqms(bqm1: BinaryQuadraticModel, bqm2: BinaryQuadraticModel) -> BinaryQuadraticModel:
    """Combine two BQMs using key-based lookup combination.
    
    This implementation correctly handles non-contiguous, negative, and slack variables
    by using variable IDs and variable_index mappings, preserving the order in which
    variables were added to avoid issues with the BQM constructor sorting variable_ids.
    
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