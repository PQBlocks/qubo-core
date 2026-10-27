"""Test for BQM combination utility function."""

import numpy as np
from pqblocks_qubo.model.bqm import BinaryQuadraticModel
from pqblocks_qubo.model.vartype import Vartype
from pqblocks_qubo.model.problems.tsp import _combine_bqms


def test_combine_bqms():
    """Test the BQM combination function."""
    # Create first BQM: H1 = 2*x0 + 3*x1 + x0*x1
    linear1 = {0: 2.0, 1: 3.0}
    quadratic1 = {(0, 1): 1.0}
    bqm1 = BinaryQuadraticModel(linear1, quadratic1, 0.0, Vartype.BINARY)
    
    # Create second BQM: H2 = x1 + 4*x2 + 2*x0*x2
    # Need to include all variables in linear dict, even if coefficient is 0
    linear2 = {0: 0.0, 1: 1.0, 2: 4.0}
    quadratic2 = {(0, 2): 2.0}
    bqm2 = BinaryQuadraticModel(linear2, quadratic2, 0.0, Vartype.BINARY)
    
    # Combine BQMs
    combined = _combine_bqms(bqm1, bqm2)
    
    # Expected result: H = 2*x0 + 4*x1 + 4*x2 + x0*x1 + 2*x0*x2
    expected_linear = {0: 2.0, 1: 4.0, 2: 4.0}
    expected_quadratic = {(0, 1): 1.0, (0, 2): 2.0}
    expected_bqm = BinaryQuadraticModel(expected_linear, expected_quadratic, 0.0, Vartype.BINARY)
    
    # Test with a sample
    sample = {0: 1, 1: 1, 2: 1}
    combined_energy = combined.energy(sample)
    expected_energy = expected_bqm.energy(sample)
    
    assert abs(combined_energy - expected_energy) < 1e-12
    
    # Test individual terms
    assert combined.num_variables == 3
    assert combined.vartype == Vartype.BINARY
    
    # Check linear terms
    assert abs(combined.linear[0] - 2.0) < 1e-12  # x0 coefficient
    assert abs(combined.linear[1] - 4.0) < 1e-12  # x1 coefficient
    assert abs(combined.linear[2] - 4.0) < 1e-12  # x2 coefficient
    
    # Check quadratic terms
    idx_0 = combined.variable_index[0]
    idx_1 = combined.variable_index[1]
    idx_2 = combined.variable_index[2]
    assert abs(combined.quadratic[idx_0, idx_1] - 1.0) < 1e-12  # x0*x1 coefficient
    assert abs(combined.quadratic[idx_0, idx_2] - 2.0) < 1e-12  # x0*x2 coefficient
    
    print("BQM combination test passed!")


if __name__ == "__main__":
    test_combine_bqms()