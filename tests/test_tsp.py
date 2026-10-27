"""Tests for TSP problem loader."""

import numpy as np
import pytest

from pqblocks_qubo.model.problems.tsp import load_tsp
from pqblocks_qubo.model.bqm import BinaryQuadraticModel
from pqblocks_qubo.model.vartype import Vartype
from pqblocks_qubo.model.constraints import compile_onehot


def test_tsp_permutation_constraints():
    """Test that TSP creates correct permutation matrix constraints."""
    # Create a simple 3-city distance matrix
    distance_matrix = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ], dtype=int)
    
    bqm, var_map = load_tsp(distance_matrix)
    
    # Verify BQM properties
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.BINARY
    assert bqm.num_variables == 9  # 3 cities × 3 time steps
    
    # Verify variable mapping
    assert len(var_map) == 9
    for v in range(3):
        for j in range(3):
            assert (v, j) in var_map
    
    # Check that we have the right number of variables
    assert set(var_map.values()) == set(range(9))


def test_tsp_4city_known_tour():
    """Test TSP with a 4-city instance and verify a known optimal tour."""
    # Create a simple 4-city distance matrix
    # This is designed to have an optimal tour 0->1->2->3->0 with distance 4
    distance_matrix = np.array([
        [0, 1, 2, 2],
        [2, 0, 1, 2],
        [2, 2, 0, 1],
        [1, 2, 2, 0]
    ], dtype=int)
    
    bqm, var_map = load_tsp(distance_matrix, A=5.0)  # Use high penalty to enforce constraints
    
    # Verify BQM properties
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.BINARY
    assert bqm.num_variables == 16  # 4 cities × 4 time steps
    
    # Create a sample representing the optimal tour: 0->1->2->3->0
    # x_{0,0} = 1, x_{1,1} = 1, x_{2,2} = 1, x_{3,3} = 1
    # All other variables = 0
    sample = {i: 0 for i in range(16)}
    sample[var_map[(0, 0)]] = 1
    sample[var_map[(1, 1)]] = 1
    sample[var_map[(2, 2)]] = 1
    sample[var_map[(3, 3)]] = 1
    
    energy = bqm.energy(sample)
    
    # Calculate expected energy:
    # Distance terms: d_{0,1} + d_{1,2} + d_{2,3} + d_{3,0} = 1 + 1 + 1 + 1 = 4
    # Constraint terms: 0 (all constraints satisfied)
    # Total = 4
    assert abs(energy - 4.0) < 1e-10


def test_tsp_auto_tuning():
    """Test that A=None triggers default penalty weighting."""
    # Create a simple 3-city distance matrix
    distance_matrix = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ], dtype=int)
    
    # Test with A=None (should use default)
    bqm_default, var_map = load_tsp(distance_matrix, A=None)
    
    # Test with explicit A=2.0 (should be the same as default)
    bqm_explicit, var_map2 = load_tsp(distance_matrix, A=2.0)
    
    # The BQMs should be equivalent
    assert bqm_default.num_variables == bqm_explicit.num_variables
    assert bqm_default.vartype == bqm_explicit.vartype
    
    # Test a simple feasible sample
    sample = {i: 0 for i in range(9)}
    # Make it feasible: each city visited once, each time step has one city
    # City 0 at time 0, City 1 at time 1, City 2 at time 2
    sample[var_map[(0, 0)]] = 1
    sample[var_map[(1, 1)]] = 1
    sample[var_map[(2, 2)]] = 1
    
    energy_default = bqm_default.energy(sample)
    energy_explicit = bqm_explicit.energy(sample)
    
    assert abs(energy_default - energy_explicit) < 1e-12


def test_tsp_empty_instance():
    """Test TSP with empty distance matrix."""
    distance_matrix = np.array([[]], dtype=int).reshape(0, 0)
    
    bqm, var_map = load_tsp(distance_matrix)
    
    # Verify BQM properties
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.BINARY
    assert bqm.num_variables == 0
    assert len(var_map) == 0
    assert bqm.offset == 0.0


def test_tsp_single_city():
    """Test TSP with single city."""
    distance_matrix = np.array([[0]], dtype=int)
    
    bqm, var_map = load_tsp(distance_matrix)
    
    # Verify BQM properties
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.BINARY
    assert bqm.num_variables == 1
    assert len(var_map) == 1
    assert (0, 0) in var_map
    assert bqm.offset == 0.0
    
    # Test the only possible sample
    sample = {var_map[(0, 0)]: 1}
    energy = bqm.energy(sample)
    assert abs(energy - 0.0) < 1e-12


def test_tsp_invalid_inputs():
    """Test TSP with invalid inputs."""
    # Test with non-square matrix
    with pytest.raises(ValueError):
        invalid_matrix = np.array([[0, 1], [1, 0], [2, 3]])
        load_tsp(invalid_matrix)
        
    # Test with 1D array
    with pytest.raises(ValueError):
        invalid_matrix = np.array([0, 1, 2])
        load_tsp(invalid_matrix)
        
    # Test with negative penalty weight
    with pytest.raises(ValueError):
        distance_matrix = np.array([[0, 1], [1, 0]])
        load_tsp(distance_matrix, A=-1.0)
        
    # Test with negative objective weight
    with pytest.raises(ValueError):
        distance_matrix = np.array([[0, 1], [1, 0]])
        load_tsp(distance_matrix, B=-1.0)


def test_tsp_constraint_satisfaction():
    """Test that valid TSP solutions satisfy all constraints."""
    # Create a 3-city distance matrix
    distance_matrix = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ], dtype=int)
    
    bqm, var_map = load_tsp(distance_matrix, A=10.0)  # High penalty to clearly see constraint violations
    
    # Create a valid TSP solution: 0->1->2->0
    valid_sample = {i: 0 for i in range(9)}
    valid_sample[var_map[(0, 0)]] = 1  # City 0 at time 0
    valid_sample[var_map[(1, 1)]] = 1  # City 1 at time 1
    valid_sample[var_map[(2, 2)]] = 1  # City 2 at time 2
    
    # This should have low energy (only distance terms)
    valid_energy = bqm.energy(valid_sample)
    
    # Create an invalid solution: two cities at the same time
    invalid_sample = {i: 0 for i in range(9)}
    invalid_sample[var_map[(0, 0)]] = 1  # City 0 at time 0
    invalid_sample[var_map[(1, 0)]] = 1  # City 1 also at time 0 (violation)
    invalid_sample[var_map[(2, 1)]] = 1  # City 2 at time 1
    
    # This should have high energy (constraint penalty)
    invalid_energy = bqm.energy(invalid_sample)
    
    # The invalid solution should have higher energy due to constraint penalties
    assert invalid_energy > valid_energy


def test_tsp_variable_mapping():
    """Test that variable mapping is consistent and complete."""
    # Create a 4-city distance matrix
    distance_matrix = np.array([
        [0, 1, 2, 3],
        [1, 0, 4, 5],
        [2, 4, 0, 6],
        [3, 5, 6, 0]
    ], dtype=int)
    
    bqm, var_map = load_tsp(distance_matrix)
    
    # Verify that the variable mapping is bijective
    # 1. All (city, time) pairs map to unique variable IDs
    assert len(var_map) == 16  # 4 × 4
    
    # 2. All variable IDs are unique and in range
    var_ids = list(var_map.values())
    assert len(set(var_ids)) == len(var_ids)  # All unique
    assert set(var_ids) == set(range(16))     # All in range 0-15
    
    # 3. All (city, time) pairs are present
    expected_keys = [(v, j) for v in range(4) for j in range(4)]
    assert set(var_map.keys()) == set(expected_keys)
    
    # 4. Check a few specific mappings
    assert var_map[(0, 0)] == 0
    assert var_map[(0, 1)] == 1
    assert var_map[(1, 0)] == 4
    assert var_map[(3, 3)] == 15