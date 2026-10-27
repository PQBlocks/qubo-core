"""Unit tests for Portfolio Optimization problem loader."""

import numpy as np
import pytest
from pqblocks_qubo.model.problems.portfolio import load_portfolio
from pqblocks_qubo.model.bqm import BinaryQuadraticModel
from pqblocks_qubo.model.vartype import Vartype


def test_portfolio_cardinality():
    """Test portfolio optimization with cardinality constraint."""
    # Create a simple covariance matrix and returns vector
    covariance = np.array([
        [0.04, 0.01, 0.02],
        [0.01, 0.09, 0.03],
        [0.02, 0.03, 0.16]
    ])
    returns = np.array([0.1, 0.2, 0.15])
    cardinality = 2
    
    # Load portfolio optimization problem
    bqm = load_portfolio(covariance, returns, cardinality)
    
    # Check that BQM is created correctly
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.BINARY
    
    # Should have 3 variables (one for each asset)
    assert bqm.num_variables == 3


def test_portfolio_with_auto_tuning():
    """Test portfolio optimization with auto-tuning of penalty weight."""
    # Create a simple covariance matrix and returns vector
    covariance = np.array([
        [0.04, 0.01],
        [0.01, 0.09]
    ])
    returns = np.array([0.1, 0.2])
    cardinality = 1
    
    # Load portfolio optimization problem with auto-tuning (A=None)
    bqm = load_portfolio(covariance, returns, cardinality, A=None)
    
    # Check that BQM is created correctly
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.BINARY


def test_portfolio_risk_appetite():
    """Test portfolio optimization with different risk appetite values."""
    # Create a simple covariance matrix and returns vector
    covariance = np.array([
        [0.04, 0.01],
        [0.01, 0.09]
    ])
    returns = np.array([0.1, 0.2])
    cardinality = 1
    
    # Test with risk appetite 0 (only return matters)
    bqm_risk_0 = load_portfolio(covariance, returns, cardinality, risk_appetite=0.0)
    
    # Test with risk appetite 1 (only risk matters)
    bqm_risk_1 = load_portfolio(covariance, returns, cardinality, risk_appetite=1.0)
    
    # Both should be valid BQMs
    assert isinstance(bqm_risk_0, BinaryQuadraticModel)
    assert isinstance(bqm_risk_1, BinaryQuadraticModel)


def test_portfolio_energy_calculation():
    """Test energy calculation for portfolio optimization."""
    # Create a simple covariance matrix and returns vector
    covariance = np.array([
        [0.04, 0.01],
        [0.01, 0.09]
    ])
    returns = np.array([0.1, 0.2])
    cardinality = 1
    
    # Load portfolio optimization problem
    bqm = load_portfolio(covariance, returns, cardinality)
    
    # Create a sample (select first asset only)
    sample = {0: 1, 1: 0}
    
    # Calculate energy
    energy = bqm.energy(sample)
    
    # Energy should be finite
    assert np.isfinite(energy)


def test_portfolio_invalid_inputs():
    """Test portfolio optimization with invalid inputs."""
    covariance = np.array([
        [0.04, 0.01],
        [0.01, 0.09]
    ])
    returns = np.array([0.1, 0.2, 0.3])  # Mismatched dimensions
    cardinality = 1
    
    # Should raise ValueError for mismatched dimensions
    with pytest.raises(ValueError):
        load_portfolio(covariance, returns, cardinality)
    
    # Should raise ValueError for invalid cardinality
    with pytest.raises(ValueError):
        load_portfolio(covariance, np.array([0.1, 0.2]), 0)
    
    # Should raise ValueError for invalid cardinality
    with pytest.raises(ValueError):
        load_portfolio(covariance, np.array([0.1, 0.2]), 3)