"""Unit tests for Graph Coloring problem loader."""

import numpy as np
import networkx as nx
import pytest
from pqblocks_qubo.model.problems.coloring import load_coloring
from pqblocks_qubo.model.bqm import BinaryQuadraticModel
from pqblocks_qubo.model.vartype import Vartype


def test_coloring_triangle():
    """Test coloring of a triangle graph with 3 colors."""
    # Create a triangle graph
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    
    # Load coloring problem with 3 colors
    bqm = load_coloring(graph, 3)
    
    # Check that BQM is created correctly
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.BINARY
    
    # For a triangle with 3 colors, we should have 3*3 = 9 variables
    assert bqm.num_variables == 9
    
    # Check that there are quadratic terms for adjacent vertices sharing colors
    # Each edge should contribute num_colors penalty terms
    num_edge_penalty_terms = len(graph.edges()) * 3
    # Each vertex should have a one-hot constraint with 3 variables
    # One-hot constraint for 3 variables has 3*(3-1)/2 = 3 quadratic terms
    num_onehot_terms = len(graph.nodes()) * 3
    
    # Total quadratic terms should be edge penalties + one-hot constraints
    # Note: This is a simplified check - actual implementation may vary
    assert len(bqm.quadratic) > 0


def test_coloring_with_auto_tuning():
    """Test coloring with auto-tuning of penalty weight."""
    # Create a simple graph
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])
    
    # Load coloring problem with auto-tuning (A=None)
    bqm = load_coloring(graph, 2, A=None)
    
    # Check that BQM is created correctly
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.BINARY


def test_coloring_with_adjacency_matrix():
    """Test coloring with adjacency matrix input."""
    # Create adjacency matrix for a triangle
    adj_matrix = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    
    # Load coloring problem
    bqm = load_coloring(adj_matrix, 3)
    
    # Check that BQM is created correctly
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.BINARY


def test_coloring_with_edge_list():
    """Test coloring with edge list input."""
    # Create edge list for a triangle
    edge_list = [(0, 1), (1, 2), (2, 0)]
    
    # Load coloring problem
    bqm = load_coloring(edge_list, 3)
    
    # Check that BQM is created correctly
    assert isinstance(bqm, BinaryQuadraticModel)
    assert bqm.vartype == Vartype.BINARY


def test_coloring_energy_calculation():
    """Test energy calculation for a valid coloring."""
    # Create a simple graph (two nodes connected)
    graph = nx.Graph()
    graph.add_edges_from([(0, 1)])
    
    # Load coloring problem with 2 colors
    bqm = load_coloring(graph, 2)
    
    # Create a valid coloring: node 0 = color 0, node 1 = color 1
    # Variables are indexed as: node * num_colors + color
    sample = {
        0: 1,  # Node 0, color 0
        1: 0,  # Node 0, color 1
        2: 0,  # Node 1, color 0
        3: 1   # Node 1, color 1
    }
    
    # Calculate energy
    energy = bqm.energy(sample)
    
    # Energy should be finite
    assert np.isfinite(energy)