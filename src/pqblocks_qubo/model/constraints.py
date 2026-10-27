"""Constraint compilation for QUBO problems."""

import numpy as np
from typing import List, Union, Tuple, Dict, Callable
from .bqm import BinaryQuadraticModel
from .vartype import Vartype


# Global counter for generating unique ancilla IDs
_ancilla_counter = 0


def _get_unique_ancilla_id() -> int:
    """Generate a unique negative ancilla ID."""
    global _ancilla_counter
    _ancilla_counter += 1
    return -_ancilla_counter


def compile_equality(variables: List[int], k: int, A: float) -> BinaryQuadraticModel:
    """Compile an equality constraint into a BinaryQuadraticModel.
    
    Creates a penalty term A * (sum(x_i) - k)^2 which is minimized when 
    exactly k variables are 1 and the rest are 0.
    
    Args:
        variables: List of variable indices. May contain duplicates.
        k: Target number of variables that should be 1.
        A: Penalty weight. Must be positive.
        
    Returns:
        BinaryQuadraticModel representing the constraint.
        
    Raises:
        ValueError: If A <= 0 or if variables list is empty after deduplication.
    """
    # Validate penalty weight
    if A <= 0:
        raise ValueError(f"Penalty weight must be positive, got {A}")
    
    # Deduplicate variables while preserving order
    seen = set()
    unique_vars = []
    for v in variables:
        if v not in seen:
            seen.add(v)
            unique_vars.append(v)
    
    # Check for empty variable list
    if len(unique_vars) == 0:
        raise ValueError("Variables list cannot be empty")
    
    n = len(unique_vars)
    
    # Calculate linear coefficients: A * (1 - 2k) for each variable
    linear_coeff = A * (1 - 2 * k)
    linear = {var: linear_coeff for var in unique_vars}
    
    # Calculate quadratic coefficients: 2A for each pair of variables
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            var_i, var_j = unique_vars[i], unique_vars[j]
            quadratic[(var_i, var_j)] = 2.0 * A
    
    # Calculate offset: A * k^2
    offset = A * k * k
    
    return BinaryQuadraticModel(linear, quadratic, offset, Vartype.BINARY)


def compile_inequality(variables: List[int], k: int, A: float) -> Tuple[BinaryQuadraticModel, List[int]]:
    """Compile a logarithmic slack inequality constraint into a BinaryQuadraticModel.
    
    Creates a penalty term A * (sum(x_i) + sum(c_j * y_j) - k)^2 which is minimized when
    sum(x_i) <= k. The slack variables y_j are used to represent the remaining capacity.
    
    Args:
        variables: List of variable indices. May contain duplicates.
        k: Upper bound on sum of variables.
        A: Penalty weight. Must be positive.
        
    Returns:
        Tuple of (BinaryQuadraticModel representing the constraint, list of slack variable indices).
        
    Raises:
        ValueError: If A <= 0, k < 0, or variables list is empty.
    """
    # Validate inputs
    if A <= 0:
        raise ValueError(f"Penalty weight must be positive, got {A}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    
    # Deduplicate variables while preserving order
    seen = set()
    unique_vars = []
    for v in variables:
        if v not in seen:
            seen.add(v)
            unique_vars.append(v)
    
    # Check for empty variable list
    if len(unique_vars) == 0:
        raise ValueError("Variables list cannot be empty")
    
    n = len(unique_vars)
    
    # Calculate M for logarithmic slack encoding
    M = int(np.floor(np.log2(k))) if k > 0 else 0
    
    # Generate slack variable coefficients
    slack_coeffs = [2**m for m in range(M + 1)]
    if M >= 0 and k > 0:
        slack_coeffs[-1] = k + 1 - 2**M
    
    # Generate unique ancilla IDs
    slack_vars = [_get_unique_ancilla_id() for _ in range(M + 1)]
    
    # Build the constraint BQM
    # Linear terms for original variables: A * (1 - 2k) * x_i
    linear = {var: A * (1 - 2 * k) for var in unique_vars}
    
    # Linear terms for slack variables: A * c_j * (1 - 2k) * y_j
    for i, coeff in enumerate(slack_coeffs):
        linear[slack_vars[i]] = A * coeff * (1 - 2 * k)
    
    # Quadratic terms for original variables: 2A * x_i * x_j
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            var_i, var_j = unique_vars[i], unique_vars[j]
            quadratic[(var_i, var_j)] = 2.0 * A
    
    # Quadratic terms between original and slack variables: 2A * c_j * x_i * y_j
    for i in range(n):
        for j, coeff in enumerate(slack_coeffs):
            var_i = unique_vars[i]
            quadratic[(var_i, slack_vars[j])] = 2.0 * A * coeff
    
    # Quadratic terms for slack variables: 2A * c_i * c_j * y_i * y_j
    for i in range(M + 1):
        for j in range(i + 1, M + 1):
            coeff_i, coeff_j = slack_coeffs[i], slack_coeffs[j]
            quadratic[(slack_vars[i], slack_vars[j])] = 2.0 * A * coeff_i * coeff_j
    
    # Offset term: A * k^2
    offset = A * k * k
    
    return BinaryQuadraticModel(linear, quadratic, offset, Vartype.BINARY), slack_vars


def compile_onehot(variables: List[int], A: float, method: str = 'standard') -> Union[BinaryQuadraticModel, Tuple[BinaryQuadraticModel, Callable]]:
    """Compile a one-hot constraint into a BinaryQuadraticModel.
    
    Creates a penalty that is minimized when exactly one variable is 1 and the rest are 0.
    
    Args:
        variables: List of variable indices. May contain duplicates.
        A: Penalty weight. Must be positive.
        method: Method to use ('standard' or 'domain_wall').
        
    Returns:
        For 'standard' method: BinaryQuadraticModel representing the constraint.
        For 'domain_wall' method: Tuple of (BinaryQuadraticModel, decoding function).
        
    Raises:
        ValueError: If A <= 0, variables list is empty, or domain_wall method is used with fewer than 2 variables.
    """
    # Validate penalty weight
    if A <= 0:
        raise ValueError(f"Penalty weight must be positive, got {A}")
    
    # Deduplicate variables while preserving order
    seen = set()
    unique_vars = []
    for v in variables:
        if v not in seen:
            seen.add(v)
            unique_vars.append(v)
    
    # Check for empty variable list
    if len(unique_vars) == 0:
        raise ValueError("Variables list cannot be empty")
    
    if method == 'standard':
        return compile_equality(unique_vars, 1, A)
    elif method == 'domain_wall':
        if len(unique_vars) < 2:
            raise ValueError("Domain-wall method requires at least 2 variables")
        
        n = len(unique_vars)
        # Generate unique ancilla IDs
        ancilla_vars = [_get_unique_ancilla_id() for _ in range(n - 1)]
        
        # Build the domain-wall constraint BQM
        # Linear terms for original variables: -A for all
        linear = {var: -A for var in unique_vars}
        
        # Linear terms for ancilla variables: A * (n - i - 1) for i-th ancilla
        for i, ancilla in enumerate(ancilla_vars):
            linear[ancilla] = A * (n - i - 1)
        
        # Quadratic terms for original variables: 2A for adjacent pairs
        quadratic = {}
        for i in range(n - 1):
            quadratic[(unique_vars[i], unique_vars[i + 1])] = 2.0 * A
        
        # Quadratic terms for ancilla variables: 2A for adjacent pairs
        for i in range(n - 2):
            quadratic[(ancilla_vars[i], ancilla_vars[i + 1])] = 2.0 * A
        
        # Quadratic terms between original and ancilla variables
        for i in range(n - 1):
            # A * x_i * y_i term
            quadratic[(unique_vars[i], ancilla_vars[i])] = A
            # -2A * x_{i+1} * y_i term
            quadratic[(unique_vars[i + 1], ancilla_vars[i])] = -2.0 * A
        
        # Offset term: 0
        offset = 0.0
        
        # Create decoding function
        def decode_domain_wall(sample: Dict[int, int]) -> Dict[int, int]:
            """Decode domain-wall encoded sample back to one-hot primary variables.
            
            Args:
                sample: Sample including both primary and ancilla variables.
                
            Returns:
                Dictionary mapping primary variables to their decoded values.
            """
            # Extract ancilla values
            ancilla_values = [sample.get(ancilla, 0) for ancilla in ancilla_vars]
            
            # Decode according to Chancellor 2019 encoding
            one_hot = {}
            if ancilla_values:
                one_hot[unique_vars[0]] = 1 - ancilla_values[0]
                for idx in range(1, len(unique_vars) - 1):
                    one_hot[unique_vars[idx]] = ancilla_values[idx-1] - ancilla_values[idx]
                one_hot[unique_vars[-1]] = ancilla_values[-1]
            else:
                # Edge case with no ancilla variables (shouldn't happen with n>=2)
                one_hot = {var: 0 for var in unique_vars}
                if unique_vars:
                    one_hot[unique_vars[0]] = 1
                    
            return one_hot
        
        return BinaryQuadraticModel(linear, quadratic, offset, Vartype.BINARY), decode_domain_wall
    else:
        raise ValueError(f"Unknown method: {method}")