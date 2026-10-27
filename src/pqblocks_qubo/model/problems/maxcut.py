"""Max-Cut problem loader."""

import numpy as np
from typing import Union, List, Tuple
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from ..bqm import BinaryQuadraticModel
from ..vartype import Vartype


def load_maxcut(graph: Union[np.ndarray, List[Tuple[int, int, float]], "nx.Graph"]) -> BinaryQuadraticModel:
    """Load a Max-Cut problem into a BinaryQuadraticModel.
    
    Supports multiple input formats:
    - NetworkX Graph with optional 'weight' edge attribute
    - Adjacency matrix as numpy array
    - Edge list as list of tuples (u, v) or (u, v, weight)
    
    The Max-Cut problem is formulated as:
    H(s) = -Σ_{(u,v)∈E} w_{uv} * (1 - s_u * s_v) / 2
    
    For SPIN variables s_u, s_v ∈ {-1, +1}, this energy is minimized when
    adjacent nodes are in different partitions (maximizing the cut value).
    
    Args:
        graph: Input graph in one of the supported formats.
        
    Returns:
        BinaryQuadraticModel with SPIN variables representing the Max-Cut problem.
        
    Raises:
        ImportError: If NetworkX is required but not installed.
        ValueError: If the input format is not supported.
    """
    linear = {}
    quadratic = {}
    offset = 0.0
    
    if HAS_NETWORKX and isinstance(graph, nx.Graph):
        # NetworkX Graph input
        n = graph.number_of_nodes()
        edges = graph.edges(data='weight', default=1.0)
        
        # Process edges and aggregate weights for duplicates/symmetric edges
        edge_weights = {}
        for u, v, weight in edges:
            # Normalize edge representation to ensure consistent key
            u, v = sorted([u, v])
            key = (u, v)
            if key in edge_weights:
                edge_weights[key] += float(weight)
            else:
                edge_weights[key] = float(weight)
                
        # Now create quadratic terms with aggregated weights
        for (u, v), weight in edge_weights.items():
            # H = -Σ w_{uv} * (1 - s_u * s_v) / 2 = -Σ w_{uv}/2 + Σ w_{uv}/2 * s_u * s_v
            quadratic[(u, v)] = 0.5 * weight  # Positive coefficient to maximize cut
            offset -= 0.5 * weight  # Negative offset to maximize cut
                
    elif isinstance(graph, np.ndarray):
        # Adjacency matrix input
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("Adjacency matrix must be 2D and square")
            
        n = graph.shape[0]
        # Process edges and aggregate weights for duplicates/symmetric edges
        edge_weights = {}
        for u in range(n):
            for v in range(u + 1, n):  # Only upper triangle to avoid double counting
                weight = graph[u, v]
                if weight != 0:
                    # Normalize edge representation to ensure consistent key
                    u_key, v_key = sorted([u, v])
                    key = (u_key, v_key)
                    if key in edge_weights:
                        edge_weights[key] += float(weight)
                    else:
                        edge_weights[key] = float(weight)
                        
        # Now create quadratic terms with aggregated weights
        for (u, v), weight in edge_weights.items():
            # H = -Σ w_{uv} * (1 - s_u * s_v) / 2 = -Σ w_{uv}/2 + Σ w_{uv}/2 * s_u * s_v
            quadratic[(u, v)] = 0.5 * weight  # Positive coefficient to maximize cut
            offset -= 0.5 * weight  # Negative offset to maximize cut
                    
    elif isinstance(graph, list):
        # Edge list input - handle both weighted and unweighted
        if not graph:
            n = 0
        else:
            # Check if first element has 3 items (weighted) or 2 items (unweighted)
            if len(graph[0]) == 3:
                n = max(max(u, v) for u, v, _ in graph) + 1
            elif len(graph[0]) == 2:
                n = max(max(u, v) for u, v in graph) + 1
            else:
                raise ValueError("Edge list elements must be tuples of length 2 or 3")
            
        # Process edges and aggregate weights for duplicates/symmetric edges
        edge_weights = {}
        for edge in graph:
            if len(edge) == 3:
                u, v, weight = edge
            elif len(edge) == 2:
                u, v = edge
                weight = 1.0
            else:
                raise ValueError("Edge list elements must be tuples of length 2 or 3")
                
            # Normalize edge representation to ensure consistent key
            u, v = sorted([u, v])
            key = (u, v)
            if key in edge_weights:
                edge_weights[key] += float(weight)
            else:
                edge_weights[key] = float(weight)
                
        # Now create quadratic terms with aggregated weights
        for (u, v), weight in edge_weights.items():
            # H = -Σ w_{uv} * (1 - s_u * s_v) / 2 = -Σ w_{uv}/2 + Σ w_{uv}/2 * s_u * s_v
            quadratic[(u, v)] = 0.5 * weight  # Positive coefficient to maximize cut
            offset -= 0.5 * weight  # Negative offset to maximize cut
            
    else:
        raise ValueError("Unsupported graph format. Must be NetworkX Graph, adjacency matrix, or edge list.")
    
    return BinaryQuadraticModel(linear, quadratic, offset, Vartype.SPIN)