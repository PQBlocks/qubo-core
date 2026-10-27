"""Problem loader for Portfolio Optimization problems."""

import numpy as np
from typing import Optional, Dict
from ..bqm import BinaryQuadraticModel
from ..vartype import Vartype
from ..constraints import compile_equality
from ..auto_tuner import PenaltyAutoTuner


def load_portfolio(
    covariance: np.ndarray,
    returns: np.ndarray,
    cardinality: int,
    risk_appetite: float = 1.0,
    A: Optional[float] = None
) -> BinaryQuadraticModel:
    """Load a Portfolio Optimization problem as a BinaryQuadraticModel.
    
    Creates a QUBO formulation for the Markowitz portfolio optimization model:
    min  risk_appetite * x^T Σ x - (1 - risk_appetite) * μ^T x
    s.t. Σx_i = cardinality
    
    where Σ is the covariance matrix, μ is the expected returns vector,
    and x is a binary vector indicating which assets are selected.
    
    Args:
        covariance: Covariance matrix of asset returns (n x n).
        returns: Expected returns vector (n).
        cardinality: Number of assets to select.
        risk_appetite: Trade-off parameter between risk and return [0, 1].
        A: Penalty weight for cardinality constraint. If None, auto-tuned.
        
    Returns:
        BinaryQuadraticModel representing the portfolio optimization problem.
    """
    n = len(returns)
    
    if covariance.shape != (n, n):
        raise ValueError("Covariance matrix must be square and match returns dimension")
    
    if cardinality <= 0 or cardinality > n:
        raise ValueError("Cardinality must be between 1 and number of assets")
    
    # Normalize covariance matrix if needed
    max_abs_cov = np.max(np.abs(covariance))
    if max_abs_cov > 1e-12:  # Avoid division by zero
        normalized_cov = covariance / max_abs_cov
    else:
        normalized_cov = covariance
    
    # Build quadratic term for risk: risk_appetite * x^T Σ x
    risk_quadratic = {}
    for i in range(n):
        for j in range(i, n):  # Only populate upper triangle
            val = risk_appetite * normalized_cov[i, j]
            if abs(val) > 1e-12:  # Only store significant values
                risk_quadratic[(i, j)] = val
    
    # Build linear term for return: -(1 - risk_appetite) * μ^T x
    return_linear = {}
    for i in range(n):
        val = -(1 - risk_appetite) * returns[i]
        if abs(val) > 1e-12:  # Only store significant values
            return_linear[i] = val
    
    # Combine risk and return terms
    # For diagonal terms, we add both risk and return components
    linear = return_linear.copy()
    quadratic = risk_quadratic.copy()
    
    # Add diagonal risk terms to linear terms
    for i in range(n):
        if (i, i) in quadratic:
            linear[i] = linear.get(i, 0.0) + quadratic[(i, i)]
            del quadratic[(i, i)]
    
    # If A is provided, we can directly construct the BQM with constraint
    if A is not None:
        # Add cardinality constraint
        constraint = compile_equality(list(range(n)), cardinality, A)
        
        # Create objective BQM
        objective = BinaryQuadraticModel(linear, quadratic, 0.0, Vartype.BINARY)
        
        # Combine objective and constraint
        return _combine_bqms(objective, constraint)
    
    # If A is None, we need to auto-tune
    # Create objective function
    objective = BinaryQuadraticModel(linear, quadratic, 0.0, Vartype.BINARY)
    
    # Define constraint function for auto-tuner
    def build_constraints(A_val: float) -> BinaryQuadraticModel:
        return compile_equality(list(range(n)), cardinality, A_val)
    
    # Define feasibility check
    def is_feasible(sample: Dict[int, int]) -> bool:
        # Check cardinality constraint
        selected_count = sum(sample.get(i, 0) for i in range(n))
        return selected_count == cardinality
    
    # Run auto-tuner
    tuner = PenaltyAutoTuner(
        objective_bqm=objective,
        constraint_bqm_func=build_constraints,
        feasibility_check=is_feasible,
        max_iters=20
    )
    tuned_A, _, _ = tuner.run()
    
    # Rebuild with tuned A
    return load_portfolio(covariance, returns, cardinality, risk_appetite, tuned_A)


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