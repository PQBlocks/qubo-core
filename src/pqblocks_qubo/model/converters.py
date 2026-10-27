"""Converters for transforming between SPIN and BINARY variable types."""

import numpy as np
from .bqm import BinaryQuadraticModel
from .vartype import Vartype


def to_qubo(bqm_ising: BinaryQuadraticModel) -> BinaryQuadraticModel:
    """Convert an Ising model to QUBO format.
    
    Args:
        bqm_ising: BinaryQuadraticModel with SPIN variables.
        
    Returns:
        BinaryQuadraticModel with BINARY variables.
        
    Raises:
        TypeError: If input is not SPIN vartype.
    """
    if bqm_ising.vartype != Vartype.SPIN:
        raise TypeError("Input must be SPIN vartype")
    
    n = bqm_ising.num_variables
    J = bqm_ising.quadratic
    h = bqm_ising.linear
    
    Q = np.zeros((n, n), dtype=np.float64)
    
    # Quadratic terms: Q_ij = 4 * J_ij (for i < j)
    # The BQM energy calculation will apply the 0.5 factor for symmetric matrices
    for i in range(n):
        for j in range(i+1, n):
            J_ij = J[i, j]
            if J_ij != 0:
                Q[i, j] = Q[j, i] = 4.0 * J_ij
    
    # Linear terms: Q_ii = 2h_i - 2 * Σ_{j≠i} J_ij
    for i in range(n):
        sum_J = 0.0
        for j in range(n):
            if i != j:
                sum_J += J[i, j]
        Q[i, i] = 2.0 * h[i] - 2.0 * sum_J
    
    # Constant offset: C = Σ_{i<j} J_ij - Σ_i h_i
    J_sum = 0.0
    for i in range(n):
        for j in range(i+1, n):
            J_sum += J[i, j]
    h_sum = np.sum(h)
    offset = bqm_ising.offset + (J_sum - h_sum)
    
    # Extract diagonal for linear terms
    Q_diag = np.diag(Q)
    
    # Set diagonal to zero in quadratic matrix as it's stored separately
    Q_no_diag = Q.copy()
    np.fill_diagonal(Q_no_diag, 0)
    
    return BinaryQuadraticModel(Q_diag, Q_no_diag, offset, Vartype.BINARY)


def to_ising(bqm_binary: BinaryQuadraticModel) -> BinaryQuadraticModel:
    """Convert a QUBO model to Ising format.
    
    Args:
        bqm_binary: BinaryQuadraticModel with BINARY variables.
        
    Returns:
        BinaryQuadraticModel with SPIN variables.
        
    Raises:
        TypeError: If input is not BINARY vartype.
    """
    if bqm_binary.vartype != Vartype.BINARY:
        raise TypeError("Input must be BINARY vartype")
    
    n = bqm_binary.num_variables
    Q = bqm_binary.quadratic
    Q_diag = bqm_binary.linear
    
    # Add diagonal back to quadratic matrix for conversion
    Q_full = Q.copy()
    np.fill_diagonal(Q_full, Q_diag)
    
    J = np.zeros((n, n), dtype=np.float64)
    h = np.zeros(n, dtype=np.float64)
    
    # Ising couplings: J_ij = Q_ij / 4 (for i < j)
    # The BQM energy calculation will apply the 0.5 factor for symmetric matrices
    for i in range(n):
        for j in range(i+1, n):
            if Q_full[i, j] != 0:
                J[i, j] = J[j, i] = Q_full[i, j] / 4.0
    
    # Ising linear terms: h_i = Q_ii/2 + Σ_j J_ij
    for i in range(n):
        sum_J = np.sum(J[i, :i]) + np.sum(J[i, i+1:])
        h[i] = Q_full[i, i] / 2.0 + sum_J
    
    # Adjust offset
    J_sum = 0.0
    for i in range(n):
        for j in range(i+1, n):
            J_sum += J[i, j]
    h_sum = np.sum(h)
    offset = bqm_binary.offset - J_sum + h_sum
    
    return BinaryQuadraticModel(h, J, offset, Vartype.SPIN)


def from_ising(linear, quadratic, offset=0.0) -> BinaryQuadraticModel:
    """Create a BinaryQuadraticModel from Ising parameters.
    
    Args:
        linear: Linear coefficients as dict or array.
        quadratic: Quadratic coefficients as dict or 2D array.
        offset: Constant offset term.
        
    Returns:
        BinaryQuadraticModel with SPIN variables.
    """
    # Accept dict or array for linear
    if isinstance(linear, dict):
        if linear:
            n = max(linear.keys()) + 1
            linear_arr = np.zeros(n)
            for i, val in linear.items():
                linear_arr[i] = val
        else:
            linear_arr = np.array([])
    else:
        linear_arr = np.asarray(linear, dtype=np.float64)
    
    # Accept dict or array for quadratic
    if isinstance(quadratic, dict):
        if quadratic:
            if isinstance(linear, dict) and not linear:
                # Infer size from quadratic dict if linear is empty
                max_var = max(max(i, j) for i, j in quadratic.keys())
                n = max_var + 1
                linear_arr = np.zeros(n, dtype=np.float64)
            elif len(linear_arr) == 0:
                # Infer size from quadratic dict if linear is empty array
                max_var = max(max(i, j) for i, j in quadratic.keys())
                n = max_var + 1
                linear_arr = np.zeros(n, dtype=np.float64)
            else:
                n = len(linear_arr)
                
            quadratic_arr = np.zeros((n, n), dtype=np.float64)
            for (i, j), val in quadratic.items():
                val_f64 = float(val)
                quadratic_arr[i, j] = quadratic_arr[j, i] = val_f64
        else:
            if len(linear_arr) == 0:
                n = 0
                quadratic_arr = np.array([]).reshape(0, 0)
            else:
                n = len(linear_arr)
                quadratic_arr = np.zeros((n, n), dtype=np.float64)
    else:
        quadratic_arr = np.asarray(quadratic, dtype=np.float64)
        if len(linear_arr) == 0:
            n = quadratic_arr.shape[0] if quadratic_arr.size > 0 else 0
            linear_arr = np.zeros(n, dtype=np.float64)
        else:
            n = len(linear_arr)
            if quadratic_arr.shape != (n, n):
                raise ValueError(f"Quadratic shape {quadratic_arr.shape} != ({n},{n})")
    
    return BinaryQuadraticModel(linear_arr, quadratic_arr, offset, Vartype.SPIN)


def from_qubo(Q, offset=0.0) -> BinaryQuadraticModel:
    """Create a BinaryQuadraticModel from QUBO parameters.
    
    Args:
        Q: Quadratic matrix as dict or 2D array.
        offset: Constant offset term.
        
    Returns:
        BinaryQuadraticModel with BINARY variables.
    """
    # Handle dict input for Q
    if isinstance(Q, dict):
        if Q:
            max_var = max(max(i, j) for i, j in Q.keys())
            n = max_var + 1
            Q_arr = np.zeros((n, n), dtype=np.float64)
            for (i, j), val in Q.items():
                val_f64 = float(val)
                Q_arr[i, j] = Q_arr[j, i] = val_f64
        else:
            n = 0
            Q_arr = np.array([]).reshape(0, 0)
    else:
        Q_arr = np.asarray(Q, dtype=np.float64)
        n = Q_arr.shape[0]
    
    # Extract diagonal for linear terms
    Q_diag = np.diag(Q_arr)
    
    # Set diagonal to zero in the quadratic matrix
    Q_no_diag = Q_arr.copy()
    np.fill_diagonal(Q_no_diag, 0)
    
    return BinaryQuadraticModel(Q_diag, Q_no_diag, offset, Vartype.BINARY)