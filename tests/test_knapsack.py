"""Unit tests for Knapsack problem loader."""

import numpy as np
import pytest
from pqblocks_qubo.model.problems.knapsack import load_knapsack
from pqblocks_qubo.model.bqm import BinaryQuadraticModel
from pqblocks_qubo.model.vartype import Vartype


def test_knapsack_small():
    """Test knapsack with a small example."""
    # Simple knapsack problem
    values = [10.0, 40.0, 30.0, 50.0]
    weights = [5.0, 4.0, 6.0, 3.0]
    capacity = 10
    
    # Load knapsack problem
    bqm = load_knapsack(values, weights, capacity)
    
    # Check that BQM is created correctly
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.BINARY
    
    # Should have 4 item variables plus slack variables
    # For capacity 10, we need floor(log2(10)) + 1 = 4 slack variables
    assert bqm.num_variables >= 4


def test_knapsack_with_auto_tuning():
    """Test knapsack with auto-tuning of penalty weight."""
    # Simple knapsack problem
    values = [10.0, 40.0, 30.0]
    weights = [5.0, 4.0, 6.0]
    capacity = 8
    
    # Load knapsack problem with auto-tuning (A=None)
    bqm = load_knapsack(values, weights, capacity, A=None)
    
    # Check that BQM is created correctly
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.BINARY


def test_knapsack_capacity_edge():
    """Test knapsack with edge cases for capacity."""
    values = [10.0, 20.0]
    weights = [5.0, 10.0]
    
    # Test with capacity 0
    bqm = load_knapsack(values, weights, 0)
    assert isinstance(bqm, BinaryQuadraticModel)
    
    # Test with capacity equal to sum of weights
    total_weight = sum(weights)
    bqm = load_knapsack(values, weights, total_weight)
    assert isinstance(bqm, BinaryQuadraticModel)


def test_knapsack_energy_calculation():
    """Test energy calculation for knapsack."""
    values = [10.0, 20.0]
    weights = [5.0, 10.0]
    capacity = 10
    
    # Load knapsack problem
    bqm = load_knapsack(values, weights, capacity)
    
    # Create a sample (select first item only)
    sample = {0: 1, 1: 0}
    
    # Calculate energy
    energy = bqm.energy(sample)
    
    # Energy should be finite
    assert np.isfinite(energy)


def test_knapsack_invalid_inputs():
    """Test knapsack with invalid inputs."""
    values = [10.0, 20.0]
    weights = [5.0, 10.0, 15.0]  # Mismatched length
    capacity = 10
    
    # Should raise ValueError for mismatched lengths
    with pytest.raises(ValueError):
        load_knapsack(values, weights, capacity)
    
    # Should raise ValueError for negative capacity
    with pytest.raises(ValueError):
        load_knapsack([10.0, 20.0], [5.0, 10.0], -5)