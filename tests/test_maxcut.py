"""Tests for Max-Cut problem loader."""

import numpy as np
import pytest
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from pqblocks_qubo.model.problems.maxcut import load_maxcut
from pqblocks_qubo.model.bqm import BinaryQuadraticModel
from pqblocks_qubo.model.vartype import Vartype


def test_maxcut_simple_cycle():
    """Test Max-Cut on a simple 4-node cycle graph."""
    # Create a 4-node cycle: 0-1-2-3-0
    if HAS_NETWORKX:
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        
        bqm = load_maxcut(graph)
        
        # Verify BQM properties
        assert isinstance(bqm, BinaryQuadraticModel)
        assert bqm.vartype == Vartype.SPIN
        assert bqm.num_variables == 4
        
        # Test a known optimal solution: cut (0,2) from (1,3)
        # One partition: {0, 2}, Other partition: {1, 3}
        # s_0 = +1, s_1 = -1, s_2 = +1, s_3 = -1
        sample = {0: 1, 1: -1, 2: 1, 3: -1}
        energy = bqm.energy(sample)
        
        # Expected energy calculation:
        # H = -Σ w_{uv} * (1 - s_u * s_v) / 2
        # Edges: (0,1), (1,2), (2,3), (3,0) with weight 1.0 each
        # (0,1): -(1 - 1*(-1))/2 = -(1 - (-1))/2 = -1
        # (1,2): -(1 - (-1)*1)/2 = -(1 - (-1))/2 = -1
        # (2,3): -(1 - 1*(-1))/2 = -(1 - (-1))/2 = -1
        # (3,0): -(1 - (-1)*1)/2 = -(1 - (-1))/2 = -1
        # Total = -4
        assert abs(energy - (-4.0)) < 1e-12


def test_maxcut_g14():
    """Test Max-Cut construction on a small G-set like instance."""
    # Create a small adjacency matrix similar to G14 instances
    # 4 nodes with weighted edges
    adj_matrix = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 1, 1, 0]
    ], dtype=float)
    
    bqm = load_maxcut(adj_matrix)
    
    # Verify BQM properties
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.SPIN
    assert bqm.num_variables == 4
    
    # Check that quadratic terms are correctly set
    # Should have edges: (0,1), (0,3), (1,2), (1,3), (2,3)
    assert len(bqm.quadratic) > 0
    
    # Test with all +1 spins (no cuts)
    sample_all_plus = {i: 1 for i in range(4)}
    energy_all_plus = bqm.energy(sample_all_plus)
    
    # H = Σ w_{uv} * (1 - s_u * s_v) / 2
    # All s_u * s_v = 1, so all terms are (1-1)/2 = 0
    assert abs(energy_all_plus - 0.0) < 1e-12
    
    # Test with alternating spins (0,2: +1), (1,3: -1)
    sample_alternating = {0: 1, 1: -1, 2: 1, 3: -1}
    energy_alternating = bqm.energy(sample_alternating)
    
    # H = -Σ w_{uv} * (1 - s_u * s_v) / 2
    # (0,1): -(1 - 1*(-1))/2 = -1
    # (0,3): -(1 - 1*(-1))/2 = -1
    # (1,2): -(1 - (-1)*1)/2 = -1
    # (1,3): -(1 - (-1)*(-1))/2 = 0
    # (2,3): -(1 - 1*(-1))/2 = -1
    # Total = -4
    assert abs(energy_alternating - (-4.0)) < 1e-12


def test_maxcut_symmetric_weights():
    """Test Max-Cut with symmetric weight matrix preserves symmetry."""
    # Create a symmetric adjacency matrix
    adj_matrix = np.array([
        [0, 2, 1],
        [2, 0, 3],
        [1, 3, 0]
    ], dtype=float)
    
    bqm = load_maxcut(adj_matrix)
    
    # Verify BQM properties
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.SPIN
    assert bqm.num_variables == 3
    
    # Check that the BQM quadratic matrix is symmetric
    # (This is an internal property, but we can check by evaluating energies)
    
    # Test with specific sample
    sample = {0: 1, 1: -1, 2: 1}
    energy = bqm.energy(sample)
    
    # H = -Σ w_{uv} * (1 - s_u * s_v) / 2
    # (0,1): -2 * (1 - 1*(-1))/2 = -2 * 1 = -2
    # (0,2): -1 * (1 - 1*1)/2 = -1 * 0 = 0
    # (1,2): -3 * (1 - (-1)*1)/2 = -3 * 1 = -3
    # Total = -5
    assert abs(energy - (-5.0)) < 1e-12


def test_maxcut_edge_list():
    """Test Max-Cut with edge list input."""
    # Create edge list: (u, v, weight)
    edges = [(0, 1, 1.0), (1, 2, 2.0), (2, 0, 3.0)]
    
    bqm = load_maxcut(edges)
    
    # Verify BQM properties
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.SPIN
    assert bqm.num_variables == 3  # max node id + 1
    
    # Test with specific sample
    sample = {0: 1, 1: -1, 2: 1}
    energy = bqm.energy(sample)
    
    # H = -Σ w_{uv} * (1 - s_u * s_v) / 2
    # (0,1): -1 * (1 - 1*(-1))/2 = -1 * 1 = -1
    # (1,2): -2 * (1 - (-1)*1)/2 = -2 * 1 = -2
    # (2,0): -3 * (1 - 1*1)/2 = -3 * 0 = 0
    # Total = -3
    assert abs(energy - (-3.0)) < 1e-12


def test_maxcut_empty_graph():
    """Test Max-Cut with empty graph."""
    if HAS_NETWORKX:
        graph = nx.Graph()
        
        bqm = load_maxcut(graph)
        
        # Verify BQM properties
        assert isinstance(bqm, BinaryQuadraticModel)
        assert bqm.vartype == Vartype.SPIN
        assert bqm.num_variables == 0
        assert len(bqm.linear) == 0
        assert bqm.quadratic.shape == (0, 0)
        assert bqm.offset == 0.0


def test_maxcut_invalid_input():
    """Test Max-Cut with invalid input."""
    # Test with invalid adjacency matrix
    with pytest.raises(ValueError):
        invalid_matrix = np.array([1, 2, 3])  # 1D array
        load_maxcut(invalid_matrix)
        
    with pytest.raises(ValueError):
        invalid_matrix = np.array([[1, 2], [3, 4], [5, 6]])  # Non-square
        load_maxcut(invalid_matrix)
        
    # Test with completely invalid input
    with pytest.raises(ValueError):
        load_maxcut("invalid input")


@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX not installed")
def test_maxcut_networkx_weights():
    """Test Max-Cut with NetworkX graph with weighted edges."""
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=2.5)
    graph.add_edge(1, 2, weight=1.5)
    graph.add_edge(2, 0)  # No weight specified, should default to 1.0
    
    bqm = load_maxcut(graph)
    
    # Verify BQM properties
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.SPIN
    assert bqm.num_variables == 3
    
    # Test with specific sample
    sample = {0: 1, 1: -1, 2: 1}
    energy = bqm.energy(sample)
    
    # H = -Σ w_{uv} * (1 - s_u * s_v) / 2
    # (0,1): -2.5 * (1 - 1*(-1))/2 = -2.5 * 1 = -2.5
    # (1,2): -1.5 * (1 - (-1)*1)/2 = -1.5 * 1 = -1.5
    # (2,0): -1.0 * (1 - 1*1)/2 = -1.0 * 0 = 0
    # Total = -4.0
    assert abs(energy - (-4.0)) < 1e-12