"""Variable types for Binary Quadratic Models."""

from enum import Enum


class Vartype(Enum):
    """Variable types for Binary Quadratic Models."""

    SPIN = 1
    """Spin variables {-1, +1}."""

    BINARY = 2
    """Binary variables {0, 1}."""