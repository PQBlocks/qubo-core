"""Unit tests for converters."""

import numpy as np
import pytest
from pqblocks_qubo.model.bqm import BinaryQuadraticModel
from pqblocks_qubo.model.vartype import Vartype
from pqblocks_qubo.model.converters import to_qubo, to_ising, from_ising, from_qubo


def test_to_qubo_energy_conservation():
    """Test energy conservation in Ising to QUBO conversion using 4-spin worked example."""
    # Ising instance from spec
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -0.5, 0.0],
        [0.0, -0.5, 0.0, 0.75],
        [0.0, 0.0, 0.75, 0.0]
    ])
    h = np.array([0.5, -0.25, 0.0, 1.0])
    
    bqm_ising = BinaryQuadraticModel(h, J, 0.0, Vartype.SPIN)
    
    # Convert to QUBO
    bqm_qubo = to_qubo(bqm_ising)
    
    # Test state s = [-1, +1, -1, +1] -> x = [0, 1, 0, 1]
    ising_sample = np.array([-1, 1, -1, 1])
    qubo_sample = np.array([0, 1, 0, 1])
    
    ising_energy = bqm_ising.energy(ising_sample)
    qubo_energy = bqm_qubo.energy(qubo_sample)
    
    # Expected: H(s) = E(x) = -1.0 (from spec worked example)
    assert abs(ising_energy - (-1.0)) < 1e-11
    assert abs(qubo_energy - (-1.0)) < 1e-11
    assert abs(ising_energy - qubo_energy) < 1e-11


def test_to_ising_roundtrip():
    """Test roundtrip conversion: Ising -> QUBO -> Ising."""
    # Create an Ising model
    h = np.array([0.5, -0.25, 0.0, 1.0])
    J = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -0.5, 0.0],
        [0.0, -0.5, 0.0, 0.75],
        [0.0, 0.0, 0.75, 0.0]
    ])
    
    bqm_ising = BinaryQuadraticModel(h, J, 0.5, Vartype.SPIN)
    
    # Convert to QUBO and back
    bqm_qubo = to_qubo(bqm_ising)
    bqm_ising_back = to_ising(bqm_qubo)
    
    # Test that coefficients are preserved
    np.testing.assert_allclose(bqm_ising_back.linear, h, atol=1e-12)
    np.testing.assert_allclose(bqm_ising_back.quadratic, J, atol=1e-12)
    assert abs(bqm_ising_back.offset - bqm_ising.offset) < 1e-12
    assert bqm_ising_back.vartype == Vartype.SPIN


def test_qubo_to_ising_roundtrip():
    """Test roundtrip conversion: QUBO -> Ising -> QUBO."""
    # Create a QUBO model based on the worked example from spec
    # Q_ii values from spec: [-1.0, -1.5, -0.5, 0.5]
    # Q_ij values from spec: Q_01=2.0, Q_12=-1.0, Q_23=1.5
    Q_diag = np.array([-1.0, -1.5, -0.5, 0.5])
    Q = np.array([
        [0.0, 2.0, 0.0, 0.0],
        [2.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.5],
        [0.0, 0.0, 1.5, 0.0]
    ])
    
    bqm_qubo = BinaryQuadraticModel(Q_diag, Q, 0.0, Vartype.BINARY)
    
    # Convert to Ising and back
    bqm_ising = to_ising(bqm_qubo)
    bqm_qubo_back = to_qubo(bqm_ising)
    
    # Test that coefficients are preserved
    np.testing.assert_allclose(bqm_qubo_back.linear, Q_diag, atol=1e-12)
    np.testing.assert_allclose(bqm_qubo_back.quadratic, Q, atol=1e-12)
    assert abs(bqm_qubo_back.offset - bqm_qubo.offset) < 1e-12
    assert bqm_qubo_back.vartype == Vartype.BINARY


def test_converter_handles_empty():
    """Test converters with empty models."""
    # Empty Ising model
    bqm_ising = BinaryQuadraticModel({}, {}, 0.0, Vartype.SPIN)
    bqm_qubo = to_qubo(bqm_ising)
    
    assert bqm_qubo.num_variables == 0
    assert bqm_qubo.offset == 0.0
    assert bqm_qubo.vartype == Vartype.BINARY
    
    # Empty QUBO model
    bqm_qubo_empty = BinaryQuadraticModel({}, {}, 0.0, Vartype.BINARY)
    bqm_ising_back = to_ising(bqm_qubo_empty)
    
    assert bqm_ising_back.num_variables == 0
    assert bqm_ising_back.offset == 0.0
    assert bqm_ising_back.vartype == Vartype.SPIN


def test_converter_single_variable():
    """Test converters with single variable models."""
    # Single variable Ising
    bqm_ising = BinaryQuadraticModel({0: 2.0}, {}, 1.5, Vartype.SPIN)
    bqm_qubo = to_qubo(bqm_ising)
    
    # For single variable: h=2.0 -> Q_00 = 2*2.0 = 4.0, offset = 1.5 - 2.0
    expected_Q_diag = np.array([4.0])
    expected_offset = 1.5 - 2.0  # original offset - sum(h)
    
    np.testing.assert_allclose(bqm_qubo.linear, expected_Q_diag, atol=1e-12)
    np.testing.assert_allclose(bqm_qubo.quadratic, np.array([[0.0]]), atol=1e-12)
    assert abs(bqm_qubo.offset - expected_offset) < 1e-12
    assert bqm_qubo.vartype == Vartype.BINARY
    
    # Convert back
    bqm_ising_back = to_ising(bqm_qubo)
    np.testing.assert_allclose(bqm_ising_back.linear, np.array([2.0]), atol=1e-12)
    assert abs(bqm_ising_back.offset - 1.5) < 1e-12
    assert bqm_ising_back.vartype == Vartype.SPIN


def test_from_ising_constructor():
    """Test from_ising constructor function."""
    # Dict input
    linear = {0: 0.5, 1: -0.25}
    quadratic = {(0, 1): 1.0}
    bqm = from_ising(linear, quadratic, 0.5)
    
    assert bqm.num_variables == 2
    assert bqm.offset == 0.5
    assert bqm.vartype == Vartype.SPIN
    np.testing.assert_allclose(bqm.linear, [0.5, -0.25], atol=1e-12)
    expected_quad = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(bqm.quadratic, expected_quad, atol=1e-12)
    
    # Array input
    linear_arr = np.array([0.5, -0.25])
    quadratic_arr = np.array([[0.0, 1.0], [1.0, 0.0]])
    bqm2 = from_ising(linear_arr, quadratic_arr, 0.5)
    
    np.testing.assert_allclose(bqm2.linear, linear_arr, atol=1e-12)
    np.testing.assert_allclose(bqm2.quadratic, quadratic_arr, atol=1e-12)
    assert bqm2.offset == 0.5
    assert bqm2.vartype == Vartype.SPIN


def test_from_qubo_constructor():
    """Test from_qubo constructor function."""
    # Dict input
    Q = {(0, 0): -1.0, (1, 1): -1.5, (0, 1): 2.0}
    bqm = from_qubo(Q, 0.5)
    
    assert bqm.num_variables == 2
    assert bqm.offset == 0.5
    assert bqm.vartype == Vartype.BINARY
    np.testing.assert_allclose(bqm.linear, [-1.0, -1.5], atol=1e-12)
    expected_quad = np.array([[0.0, 2.0], [2.0, 0.0]])
    np.testing.assert_allclose(bqm.quadratic, expected_quad, atol=1e-12)
    
    # Array input - diagonal elements should be extracted to linear terms
    Q_arr = np.array([[-1.0, 2.0], [2.0, -1.5]])
    bqm2 = from_qubo(Q_arr, 0.5)
    
    np.testing.assert_allclose(bqm2.linear, [-1.0, -1.5], atol=1e-12)
    expected_quad_arr = np.array([[0.0, 2.0], [2.0, 0.0]])
    np.testing.assert_allclose(bqm2.quadratic, expected_quad_arr, atol=1e-12)
    assert bqm2.offset == 0.5
    assert bqm2.vartype == Vartype.BINARY


def test_converter_wrong_vartype():
    """Test that converters raise TypeError for wrong vartype."""
    # Create BINARY model and try to convert with to_qubo
    bqm_binary = BinaryQuadraticModel({0: 1.0}, {}, 0.0, Vartype.BINARY)
    with pytest.raises(TypeError, match="Input must be SPIN vartype"):
        to_qubo(bqm_binary)
    
    # Create SPIN model and try to convert with to_ising
    bqm_spin = BinaryQuadraticModel({0: 1.0}, {}, 0.0, Vartype.SPIN)
    with pytest.raises(TypeError, match="Input must be BINARY vartype"):
        to_ising(bqm_spin)


def test_energy_conservation_all_configs():
    """Test energy conservation for all possible configurations of 2-variable system."""
    # Create a simple 2-variable Ising model
    h = np.array([0.5, -0.3])
    J = np.array([[0.0, 1.2], [1.2, 0.0]])
    bqm_ising = BinaryQuadraticModel(h, J, 0.1, Vartype.SPIN)
    
    # Convert to QUBO
    bqm_qubo = to_qubo(bqm_ising)
    
    # Test all 4 possible spin configurations
    spin_configs = [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ]
    
    for spin_sample in spin_configs:
        # Convert spin sample to binary
        binary_sample = [(s + 1) // 2 for s in spin_sample]
        
        # Calculate energies
        ising_energy = bqm_ising.energy(spin_sample)
        qubo_energy = bqm_qubo.energy(binary_sample)
        
        # They should be equal
        assert abs(ising_energy - qubo_energy) < 1e-11, \
            f"Energy mismatch for sample {spin_sample}: Ising={ising_energy}, QUBO={qubo_energy}"