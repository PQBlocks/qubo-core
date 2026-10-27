"""
PQBlocks QUBO: Core QUBO and Ising models for quantum-inspired classical optimization.
"""

from .model.bqm import BinaryQuadraticModel
from .model.vartype import Vartype
from .model.converters import to_qubo, to_ising, from_ising, from_qubo
from .model.constraints import compile_equality, compile_inequality, compile_onehot
from .model.auto_tuner import PenaltyAutoTuner
from .model.problems.maxcut import load_maxcut
from .model.problems.tsp import load_tsp
from .model.problems.coloring import load_coloring
from .model.problems.knapsack import load_knapsack
from .model.problems.portfolio import load_portfolio

__all__ = [
    "BinaryQuadraticModel",
    "Vartype",
    "to_qubo",
    "to_ising",
    "from_ising",
    "from_qubo",
    "compile_equality",
    "compile_inequality",
    "compile_onehot",
    "PenaltyAutoTuner",
    "load_maxcut",
    "load_tsp",
    "load_coloring",
    "load_knapsack",
    "load_portfolio",
]
