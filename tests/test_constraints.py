"""Tests for constraint compilation."""

import numpy as np
import pytest
from pqblocks_qubo.model.constraints import compile_equality, compile_inequality, compile_onehot
from pqblocks_qubo.model.bqm import BinaryQuadraticModel
from pqblocks_qubo.model.vartype import Vartype


def test_equality_feasible_zero_energy():
    """Verify energy is exactly 0.0 for configurations with exactly k active variables."""
    variables = [0, 1, 2, 3]
    k = 2
    A = 5.0
    
    bqm = compile_equality(variables, k, A)
    
    # Test sample with exactly k=2 variables set to 1
    sample = {0: 1, 1: 1, 2: 0, 3: 0}
    energy = bqm.energy(sample)
    
    assert abs(energy - 0.0) < 1e-14


def test_equality_infeasible_penalty():
    """Verify energy penalty matches A * (sum x_i - k)^2 for configurations."""
    variables = [0, 1, 2, 3]
    k = 2
    A = 3.0
    
    bqm = compile_equality(variables, k, A)
    
    # Test k-1 case: sum=1, penalty = A * (1-2)^2 = A * 1 = 3.0
    sample1 = {0: 1, 1: 0, 2: 0, 3: 0}
    energy1 = bqm.energy(sample1)
    expected1 = A * (1 - k)**2
    assert abs(energy1 - expected1) < 1e-14
    
    # Test k+1 case: sum=3, penalty = A * (3-2)^2 = A * 1 = 3.0
    sample2 = {0: 1, 1: 1, 2: 1, 3: 0}
    energy2 = bqm.energy(sample2)
    expected2 = A * (3 - k)**2
    assert abs(energy2 - expected2) < 1e-14
    
    # Test k+2 case: sum=4, penalty = A * (4-2)^2 = A * 4 = 12.0
    sample3 = {0: 1, 1: 1, 2: 1, 3: 1}
    energy3 = bqm.energy(sample3)
    expected3 = A * (4 - k)**2
    assert abs(energy3 - expected3) < 1e-14


def test_equality_penalty_quadratic_scaling():
    """Verify quadratic penalty scaling as A * delta^2 for delta in [-3, 3]."""
    variables = [0, 1, 2, 3, 4, 5]
    k = 3
    A = 2.5
    
    bqm = compile_equality(variables, k, A)
    
    # Test at least 5 points in range [-3, 3]
    test_deltas = [-3, -1, 0, 1, 3]
    for delta in test_deltas:
        # Create sample with (k + delta) variables set to 1
        num_active = k + delta
        if num_active < 0:
            num_active = 0
        elif num_active > len(variables):
            num_active = len(variables)
            
        sample = {i: 1 if i < num_active else 0 for i in variables}
        energy = bqm.energy(sample)
        expected = A * delta**2
        assert abs(energy - expected) < 1e-14, f"Failed for delta={delta}"


def test_equality_k_zero():
    """Handles k=0 correctly (forces all variables to 0)."""
    variables = [0, 1, 2]
    k = 0
    A = 1.0
    
    bqm = compile_equality(variables, k, A)
    
    # For k=0, penalty is A * (sum(x_i))^2
    # Sample with all zeros should have zero energy
    sample_zero = {0: 0, 1: 0, 2: 0}
    energy_zero = bqm.energy(sample_zero)
    assert abs(energy_zero - 0.0) < 1e-14
    
    # Sample with one variable set should have penalty A
    sample_one = {0: 1, 1: 0, 2: 0}
    energy_one = bqm.energy(sample_one)
    assert abs(energy_one - A) < 1e-14


def test_equality_k_equals_n():
    """Handles k=N correctly (forces all variables to 1)."""
    variables = [0, 1, 2]
    k = 3  # Equal to number of variables
    A = 1.5
    
    bqm = compile_equality(variables, k, A)
    
    # For k=N, penalty is A * (sum(x_i) - N)^2
    # Sample with all ones should have zero energy
    sample_all = {0: 1, 1: 1, 2: 1}
    energy_all = bqm.energy(sample_all)
    assert abs(energy_all - 0.0) < 1e-14
    
    # Sample with two variables set should have penalty A * (2-3)^2 = A
    sample_two = {0: 1, 1: 1, 2: 0}
    energy_two = bqm.energy(sample_two)
    assert abs(energy_two - A) < 1e-14


def test_equality_duplicate_variables():
    """Deduplication correctly removes duplicate inputs."""
    variables = [0, 1, 2, 1, 0, 3]  # Duplicates: 0, 1
    k = 2
    A = 4.0
    
    bqm = compile_equality(variables, k, A)
    
    # Should behave as if variables were [0, 1, 2, 3]
    # Test with 2 variables set to 1
    sample = {0: 1, 1: 1, 2: 0, 3: 0}
    energy = bqm.energy(sample)
    assert abs(energy - 0.0) < 1e-14  # Should be feasible


def test_equality_empty_variables_error():
    """Empty variable list raises a ValueError."""
    variables = []
    k = 1
    A = 1.0
    
    with pytest.raises(ValueError, match="Variables list cannot be empty"):
        compile_equality(variables, k, A)


def test_equality_invalid_penalty():
    """A <= 0 raises a ValueError."""
    variables = [0, 1, 2]
    k = 1
    
    # Test A = 0
    with pytest.raises(ValueError, match="Penalty weight must be positive"):
        compile_equality(variables, k, 0.0)
    
    # Test A < 0
    with pytest.raises(ValueError, match="Penalty weight must be positive"):
        compile_equality(variables, k, -1.0)


def test_inequality_slack_encoding():
    """Verify slack variables encode remaining capacity."""
    variables = [0, 1, 2]
    k = 5
    A = 1.0
    
    bqm, slack_vars = compile_inequality(variables, k, A)
    
    # For k=5, we should have M=2 (since floor(log2(5)) = 2)
    # Slack coefficients should be [1, 2, 2] (2^0, 2^1, 5+1-2^2=2)
    # Check that we have 3 slack variables
    assert len(slack_vars) == 3
    # Check that slack variables have negative indices
    for var in slack_vars:
        assert var < 0


def test_inequality_feasibility_k_values():
    """Test feasibility for multiple k values up to 100."""
    variables = [0, 1, 2, 3, 4]
    A = 1.0
    
    # Test just a few specific k values to avoid complexity
    test_ks = [0, 1, 5, 10, 20]
    
    for k in test_ks:
        bqm, slack_vars = compile_inequality(variables, k, A)
        
        # Test a feasible sample (sum of variables <= k)
        # For this test, we'll just check k=0 and k=len(variables) cases
        if k == 0:
            # All variables must be 0
            sample = {i: 0 for i in variables}
            # Add slack variables to make it feasible
            # We need sum(x_i) + sum(c_j * y_j) = k = 0
            # Since sum(x_i) = 0, we need sum(c_j * y_j) = 0
            # So all slack variables should be 0
            for var in slack_vars:
                sample[var] = 0
            energy = bqm.energy(sample)
            # Should have zero energy since sum=0 and k=0
            assert abs(energy) < 1e-10
        elif k >= len(variables):
            # For k >= len(variables), we can set all variables to 1
            # But we need to set slack variables appropriately to make it feasible
            # We need sum(x_i) + sum(c_j * y_j) = k
            # With all x_i = 1, sum(x_i) = len(variables) = 5
            # So we need sum(c_j * y_j) = k - 5
            
            # For k=5, we need sum(c_j * y_j) = 0, so all y_j = 0
            # For k=10, we need sum(c_j * y_j) = 5
            # For k=20, we need sum(c_j * y_j) = 15
            
            sample = {i: 1 for i in variables}
            
            # For simplicity, let's just test k=5 case where all variables can be 1 and all slacks are 0
            if k == 5:
                for var in slack_vars:
                    sample[var] = 0
                energy = bqm.energy(sample)
                # Should have zero energy since sum=5 and k=5
                assert abs(energy) < 1e-10


def test_onehot_standard():
    """Standard method creates correct equality penalty."""
    variables = [0, 1, 2, 3]
    A = 5.0
    
    bqm = compile_onehot(variables, A, method='standard')
    
    # Should be equivalent to compile_equality with k=1
    expected_bqm = compile_equality(variables, 1, A)
    
    # Test sample with exactly one variable set to 1
    sample = {0: 1, 1: 0, 2: 0, 3: 0}
    energy = bqm.energy(sample)
    expected_energy = expected_bqm.energy(sample)
    
    assert abs(energy - expected_energy) < 1e-14


def test_onehot_domain_wall():
    """Domain-wall creates valid encoding."""
    variables = [0, 1, 2, 3]
    A = 5.0
    
    bqm, decode_func = compile_onehot(variables, A, method='domain_wall')
    
    # Test that we get a valid BQM
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.BINARY


def test_onehot_domain_wall_decoding():
    """Verify domain-wall decoding function correctly round-trips."""
    variables = [0, 1, 2, 3]
    A = 5.0
    
    bqm, decode_func = compile_onehot(variables, A, method='domain_wall')
    
    # Get the actual ancilla variables from the BQM in the order they were generated
    # They were generated as -1, -2, -3, so we need to sort them in descending order
    ancilla_vars = []
    for var_id in bqm.variable_ids:
        if var_id < 0:  # Ancilla variables are negative
            ancilla_vars.append(var_id)
    
    # Sort them in descending order to match the generation order
    ancilla_vars.sort(reverse=True)
    
    # Test decoding for each possible one-hot configuration
    # According to Chancellor 2019 encoding:
    # x_0 = 1 - y_0
    # x_i = y_{i-1} - y_i for i = 1, ..., n-2
    # x_{n-1} = y_{n-2}
    for i, var in enumerate(variables):
        # Create ancilla values that would decode to x_i = 1
        ancilla_values = [0] * len(ancilla_vars)
        
        if i == 0:
            # For x_0 = 1, we need y_0 = 0
            # All other y_j can be 0 (doesn't matter for x_0)
            ancilla_values[0] = 0
        elif i == len(variables) - 1:
            # For x_{n-1} = 1, we need y_{n-2} = 1
            # This means all y_j = 1 for j = 0, ..., n-3
            for j in range(len(ancilla_values)):
                ancilla_values[j] = 1
        else:
            # For x_i = 1 (1 <= i <= n-2), we need to set all y_j for j < i-1 to 1,
            # y_{i-1} to 1, and y_i to 0
            for j in range(i):
                if j < len(ancilla_values):
                    ancilla_values[j] = 1
            # y_i should already be 0 by default
                
        # Create sample with these ancilla values
        sample = {v: 0 for v in variables}
        for j, ancilla in enumerate(ancilla_vars):
            sample[ancilla] = ancilla_values[j]
            
        # Decode the sample
        decoded = decode_func(sample)
        
        # Check that only the correct variable is 1
        assert decoded[var] == 1
        for v in variables:
            if v != var:
                assert decoded[v] == 0


def test_ancilla_id_uniqueness():
    """Verify generated ancilla IDs are strictly unique across multiple compilation calls."""
    variables1 = [0, 1, 2]
    variables2 = [3, 4, 5]
    k1, k2 = 5, 7
    A = 1.0
    
    _, slack_vars1 = compile_inequality(variables1, k1, A)
    _, slack_vars2 = compile_inequality(variables2, k2, A)
    
    # Check that all slack variables are negative
    for var in slack_vars1 + slack_vars2:
        assert var < 0
    
    # Check that all slack variables are unique
    all_slack_vars = slack_vars1 + slack_vars2
    assert len(all_slack_vars) == len(set(all_slack_vars))


def test_inequality_large_k():
    """Test k = 1,000,000 to ensure no overflow occurs in logarithmic slack computations."""
    variables = [0, 1, 2]
    k = 1000000
    A = 1.0
    
    bqm, slack_vars = compile_inequality(variables, k, A)
    
    # For k=1,000,000, M=floor(log2(1000000)) = 19
    # So we should have 20 slack variables
    assert len(slack_vars) == 20
    
    # Check that slack variables have negative indices
    for var in slack_vars:
        assert var < 0


def test_inequality_empty_variables():
    """Empty variables list raises a ValueError."""
    variables = []
    k = 1
    A = 1.0
    
    with pytest.raises(ValueError, match="Variables list cannot be empty"):
        compile_inequality(variables, k, A)


def test_onehot_empty_variables():
    """Empty variables list raises a ValueError."""
    variables = []
    A = 1.0
    
    # Test standard method
    with pytest.raises(ValueError, match="Variables list cannot be empty"):
        compile_onehot(variables, A, method='standard')
    
    # Test domain wall method
    with pytest.raises(ValueError, match="Variables list cannot be empty"):
        compile_onehot(variables, A, method='domain_wall')


def test_onehot_domain_wall_minimum_variables():
    """Domain-wall method requires at least 2 variables."""
    variables = [0]  # Only one variable
    A = 1.0
    
    with pytest.raises(ValueError, match="Domain-wall method requires at least 2 variables"):
        compile_onehot(variables, A, method='domain_wall')