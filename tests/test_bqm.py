"""Unit tests for BinaryQuadraticModel."""

import numpy as np
import pytest
from pqblocks_qubo.model.bqm import BinaryQuadraticModel
from pqblocks_qubo.model.vartype import Vartype


def test_bqm_construct_dict_input():
    """Test constructing BQM from dictionary inputs."""
    linear = {0: 1.0, 1: -2.0}
    quadratic = {(0, 1): 0.5}
    offset = 1.5
    bqm = BinaryQuadraticModel(linear, quadratic, offset, Vartype.BINARY)
    
    assert bqm.num_variables == 2
    assert bqm.offset == 1.5
    assert bqm.vartype == Vartype.BINARY
    np.testing.assert_array_equal(bqm.linear, [1.0, -2.0])
    expected_quad = np.array([[0.0, 0.5], [0.5, 0.0]])
    np.testing.assert_array_equal(bqm.quadratic, expected_quad)


def test_bqm_construct_array_input():
    """Test constructing BQM from array inputs."""
    linear = np.array([1.0, -2.0])
    quadratic = np.array([[0.0, 0.5], [0.5, 0.0]])
    offset = 1.5
    bqm = BinaryQuadraticModel(linear, quadratic, offset, Vartype.SPIN)
    
    assert bqm.num_variables == 2
    assert bqm.offset == 1.5
    assert bqm.vartype == Vartype.SPIN
    np.testing.assert_array_equal(bqm.linear, [1.0, -2.0])
    np.testing.assert_array_equal(bqm.quadratic, quadratic)


def test_4spin_worked_example():
    """Test the 4-spin worked example from the specification."""
    # Ising instance from spec
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -0.5, 0.0],
        [0.0, -0.5, 0.0, 0.75],
        [0.0, 0.0, 0.75, 0.0]
    ])
    h = np.array([0.5, -0.25, 0.0, 1.0])
    
    bqm = BinaryQuadraticModel(h, J, 0.0, Vartype.SPIN)
    
    # Test state s = [-1, +1, -1, +1]
    sample = np.array([-1, 1, -1, 1])
    energy = bqm.energy(sample)
    
    # Expected: H(s) = -1.0 (from spec worked example)
    assert abs(energy - (-1.0)) < 1e-12


def test_energies_vectorized():
    """Test vectorized energy calculation."""
    linear = np.array([1.0, -1.0])
    quadratic = np.array([[0.0, 0.5], [0.5, 0.0]])
    bqm = BinaryQuadraticModel(linear, quadratic, 0.0, Vartype.BINARY)
    
    # Test all possible 2-variable binary states
    samples = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    energies = bqm.energies(samples)
    expected = np.array([
        0.0,  # 0*1 + 0*(-1) + 0.5*0*0 = 0
        -1.0, # 0*1 + 1*(-1) + 0.5*0*1 = -1
        1.0,  # 1*1 + 0*(-1) + 0.5*1*0 = 1
        0.5   # 1*1 + 1*(-1) + 0.5*1*1 = 0.5
    ])
    
    np.testing.assert_allclose(energies, expected, atol=1e-12)


def test_energy_single_sample():
    """Test single sample energy calculation."""
    linear = np.array([1.0, -1.0])
    quadratic = np.array([[0.0, 0.5], [0.5, 0.0]])
    bqm = BinaryQuadraticModel(linear, quadratic, 0.0, Vartype.BINARY)
    
    sample = np.array([1, 0])
    energy = bqm.energy(sample)
    expected = 1.0  # 1*1 + 0*(-1) + 0.5*1*0 = 1
    
    assert abs(energy - expected) < 1e-12


def test_empty_bqm():
    """Test BQM with no variables."""
    linear = {}
    quadratic = {}
    bqm = BinaryQuadraticModel(linear, quadratic, 0.0, Vartype.BINARY)
    
    assert bqm.num_variables == 0
    assert bqm.offset == 0.0
    np.testing.assert_array_equal(bqm.linear, np.array([]))
    np.testing.assert_array_equal(bqm.quadratic, np.array([]).reshape(0, 0))


def test_single_variable():
    """Test BQM with single variable."""
    linear = {0: 2.0}
    quadratic = {}
    bqm = BinaryQuadraticModel(linear, quadratic, 1.5, Vartype.SPIN)
    
    assert bqm.num_variables == 1
    assert bqm.offset == 1.5
    np.testing.assert_array_equal(bqm.linear, [2.0])
    np.testing.assert_array_equal(bqm.quadratic, np.array([[0.0]]))
    
    # Test energies
    assert abs(bqm.energy([1]) - (2.0 * 1 + 1.5)) < 1e-12
    assert abs(bqm.energy([-1]) - (2.0 * (-1) + 1.5)) < 1e-12


def test_disconnected_graph():
    """Test BQM with disconnected variables."""
    linear = {0: 1.0, 1: -1.0, 2: 0.5}
    quadratic = {}  # No connections
    bqm = BinaryQuadraticModel(linear, quadratic, 0.0, Vartype.BINARY)
    
    sample = np.array([1, 0, 1])
    energy = bqm.energy(sample)
    expected = 1.0 * 1 + (-1.0) * 0 + 0.5 * 1 + 0.0  # linear terms only
    
    assert abs(energy - expected) < 1e-12


def test_symmetry_validation():
    """Test that asymmetric quadratic matrices raise ValueError."""
    linear = np.array([1.0, -1.0])
    # Asymmetric matrix
    quadratic = np.array([[0.0, 0.5], [0.3, 0.0]])
    
    with pytest.raises(ValueError, match="Quadratic matrix must be symmetric"):
        BinaryQuadraticModel(linear, quadratic, 0.0, Vartype.BINARY)


def test_coefficient_bounds():
    """Test that coefficients exceeding bounds raise OverflowError."""
    # Test large linear coefficient
    linear = {0: 1e13}
    quadratic = {}
    
    with pytest.raises(OverflowError, match="Coefficient magnitude exceeds 1e12"):
        BinaryQuadraticModel(linear, quadratic, 0.0, Vartype.BINARY)
    
    # Test large quadratic coefficient
    linear = {0: 1.0, 1: -1.0}
    quadratic = {(0, 1): 1e13}
    
    with pytest.raises(OverflowError, match="Coefficient magnitude exceeds 1e12"):
        BinaryQuadraticModel(linear, quadratic, 0.0, Vartype.BINARY)


def test_copy_independence():
    """Test that copy() creates independent BQM."""
    linear = {0: 1.0, 1: -2.0}
    quadratic = {(0, 1): 0.5}
    bqm1 = BinaryQuadraticModel(linear, quadratic, 1.5, Vartype.BINARY)
    
    bqm2 = bqm1.copy()
    
    # Modify original
    bqm1.linear[0] = 5.0
    bqm1.quadratic[0, 1] = 2.0
    bqm1.offset = 10.0
    
    # Check copy is unchanged
    assert bqm2.linear[0] == 1.0
    assert bqm2.quadratic[0, 1] == 0.5
    assert bqm2.offset == 1.5
    assert bqm2.vartype == Vartype.BINARY


def test_sample_dimension_mismatch():
    """Test that mismatched sample dimensions raise ValueError."""
    linear = np.array([1.0, -1.0])
    quadratic = np.array([[0.0, 0.5], [0.5, 0.0]])
    bqm = BinaryQuadraticModel(linear, quadratic, 0.0, Vartype.BINARY)
    
    # Test energy with wrong dimension
    with pytest.raises(ValueError, match="Sample length mismatch"):
        bqm.energy([1, 0, 1])  # 3 elements for 2-variable model
    
    # Test energies with wrong dimension
    with pytest.raises(ValueError, match="Sample dimension 3 != 2"):
        bqm.energies(np.array([[1, 0, 1]]))


def test_to_dense():
    """Test to_dense method."""
    linear = {0: 1.0, 1: -2.0}
    quadratic = {(0, 1): 0.5}
    bqm = BinaryQuadraticModel(linear, quadratic, 0.0, Vartype.BINARY)
    
    dense = bqm.to_dense()
    expected = np.array([[0.0, 0.5], [0.5, 0.0]])
    np.testing.assert_array_equal(dense, expected)
    
    # Verify it's a copy (modifying doesn't affect original)
    dense[0, 1] = 2.0
    assert bqm.quadratic[0, 1] == 0.5  # Original unchanged