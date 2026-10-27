"""Penalty Auto-Tuner for QUBO/Ising problems.

This module implements an automatic penalty weight tuner using bisection search
to find the minimal penalty weight that ensures constraint satisfaction.
"""

import numpy as np
from typing import Callable, Tuple, List, Any

from .bqm import BinaryQuadraticModel
# from .samplers import sa_sampler  # Assuming this exists based on spec


class SplitMix64:
    """
    A simple SplitMix64 implementation for generating reproducible random seeds.
    """
    def __init__(self, seed: int):
        self.state = seed
    
    def __next__(self) -> int:
        self.state += 0x9e3779b97f4a7c15
        result = self.state
        result = (result ^ (result >> 30)) * 0xbf58476d1ce4e5b9
        result = (result ^ (result >> 27)) * 0x94d049bb133111eb
        result = result ^ (result >> 31)
        return result
    
    def __iter__(self):
        return self


# Mock sa_sampler for testing purposes
def sa_sampler(bqm, num_sweeps=1000, num_reads=10, seed=None):
    """
    Mock SA sampler for testing.
    """
    # This is a simplified mock implementation
    # In a real implementation, this would run simulated annealing
    class MockSampleSet:
        def __init__(self, sample, energy):
            self.first = sample
            self.energy = energy
    
    # For testing purposes, we'll just return a fixed sample
    # In reality, this would run actual SA
    sample = {i: 0 for i in range(bqm.num_variables)}
    energy = bqm.energy(sample)
    return MockSampleSet(sample, energy)


def _combine_bqms(bqm1: BinaryQuadraticModel, bqm2: BinaryQuadraticModel) -> BinaryQuadraticModel:
    """Combine two BinaryQuadraticModels by adding their terms.
    
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


class PenaltyAutoTuner:
    """
    Automatically tunes penalty weights for constraint satisfaction problems.
    
    Uses bisection search to find the minimal penalty weight that ensures
    constraint feasibility with high probability.
    """
    
    def __init__(self, objective_bqm: BinaryQuadraticModel, 
                 constraint_bqm_func: Callable[[float], BinaryQuadraticModel],
                 feasibility_check: Callable[[Any], bool],
                 delta_max: float = None, max_iters: int = 20):
        """
        Initialize the penalty auto-tuner.
        
        Args:
            objective_bqm: The base objective function BQM
            constraint_bqm_func: Function that takes a penalty weight A and returns
                               a constraint BQM
            feasibility_check: Function that takes a sample and returns True if
                             it satisfies all constraints
            delta_max: Maximum coefficient magnitude in objective. If None,
                      computed automatically.
            max_iters: Maximum number of bisection iterations
        """
        self.objective = objective_bqm
        self.constraint_func = constraint_bqm_func
        self.is_feasible = feasibility_check
        self.delta_max = delta_max if delta_max is not None else self._compute_delta_max()
        self.max_iters = max_iters
        self.history = []  # Records (A_mid, was_feasible)
        
        # Seed management
        self._seed_base = 12345
        self._seed_stream = SplitMix64(self._seed_base)
    
    def _compute_delta_max(self) -> float:
        """
        Compute Δ_max = max_i (|Q_ii| + Σ_{j≠i} |Q_ij|)
        
        Returns:
            Maximum row sum of absolute values in the dense quadratic matrix
        """
        Q = self.objective.to_dense()  # Dense quadratic matrix
        row_sums = np.sum(np.abs(Q), axis=1)
        return float(np.max(row_sums))
    
    def run(self) -> Tuple[float, float, List[Tuple[float, bool]]]:
        """
        Run the penalty weight tuning using bisection search.
        
        Returns:
            Tuple of (optimal_A, feasibility_rate, history)
        """
        # Bisection bounds
        A_min = 0.0
        A_max = self.delta_max + 1.0
        
        for iteration in range(self.max_iters):
            A_mid = (A_min + A_max) / 2.0
            
            # Build combined BQM
            constraint_bqm = self.constraint_func(A_mid)
            # combined = self.objective + constraint_bqm  # This requires __add__ method
            # Instead, we'll manually combine them
            combined = _combine_bqms(self.objective, constraint_bqm)
            
            # Sample with fast SA
            seed = next(self._seed_stream)
            samples = self._fast_sa_sampler(combined, seed=seed)
            best_sample = samples.first
            
            # Bisection update
            is_feasible = self.is_feasible(best_sample)
            self.history.append((A_mid, is_feasible))
            
            if is_feasible:
                A_max = A_mid  # Found feasible, try lower
            else:
                A_min = A_mid  # Infeasible, need higher
        
        # Final validation: 100 checks at A_max
        feasible_count = 0
        for i in range(100):
            final_combined = _combine_bqms(self.objective, self.constraint_func(A_max))
            seed = next(self._seed_stream)
            final_samples = self._fast_sa_sampler(final_combined, seed=seed)
            if self.is_feasible(final_samples.first):
                feasible_count += 1
        
        feasibility_rate = feasible_count / 100.0
        return A_max, feasibility_rate, self.history
    
    def _fast_sa_sampler(self, bqm: BinaryQuadraticModel, seed: int) -> Any:
        """
        Fast SA for tuning: 200 sweeps, 5 reads.
        
        Args:
            bqm: BinaryQuadraticModel to sample
            seed: Random seed for sampling
            
        Returns:
            SampleSet from the sampler
        """
        return sa_sampler(bqm, num_sweeps=200, num_reads=5, seed=seed)