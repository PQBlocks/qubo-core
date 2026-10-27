"""Binary Quadratic Model implementation."""

import numpy as np
from typing import Union, Dict, Tuple, Any
from .vartype import Vartype


class BinaryQuadraticModel:
    """A Binary Quadratic Model (BQM) representing a quadratic polynomial over binary variables.

    The BQM can be represented as:
    E(x) = x^T Q x + offset (for BINARY)
    H(s) = s^T J s + h^T s (for SPIN)

    Supports both SPIN and BINARY variable types.
    """

    def __init__(
        self,
        linear: Union[Dict[int, float], np.ndarray],
        quadratic: Union[Dict[Tuple[int, int], float], np.ndarray],
        offset: float,
        vartype: Vartype,
    ):
        """Initialize a BinaryQuadraticModel.

        Args:
            linear: Linear coefficients as dict {variable: coefficient} or array.
            quadratic: Quadratic coefficients as dict {(i, j): coefficient} or 2D array.
            offset: Constant offset term.
            vartype: Variable type (SPIN or BINARY).

        Raises:
            ValueError: If quadratic matrix is not symmetric or dimensions don't match.
            OverflowError: If coefficient magnitude exceeds 1e12.
        """
        # Handle dict input for linear
        if isinstance(linear, dict):
            if linear:
                # Create a mapping from variable IDs to contiguous indices
                # Preserve the order in which variables were added to avoid issues with the BQM constructor sorting variable_ids
                self.variable_ids = list(linear.keys())  # Keep the original order
                self.variable_index = {var_id: idx for idx, var_id in enumerate(self.variable_ids)}
                n = len(self.variable_ids)
                self.linear = np.zeros(n, dtype=np.float64)
                for var_id, val in linear.items():
                    idx = self.variable_index[var_id]
                    self.linear[idx] = float(val)
            else:
                self.linear = np.array([], dtype=np.float64)
                self.variable_ids = []
                self.variable_index = {}
                n = 0
        else:
            self.linear = np.asarray(linear, dtype=np.float64).flatten()
            n = len(self.linear)
            # For array input, assume variable IDs are 0, 1, 2, ..., n-1
            self.variable_ids = list(range(n))
            self.variable_index = {var_id: idx for idx, var_id in enumerate(self.variable_ids)}

        # Quadratic: accept dict or array, always convert to dense for internal
        if isinstance(quadratic, dict):
            if n == 0 and quadratic:
                # Infer size and variable IDs from quadratic dict if linear is empty
                # Preserve the order in which variables were added to avoid issues with the BQM constructor sorting variable_ids
                all_vars = []
                seen = set()
                for i, j in quadratic.keys():
                    if i not in seen:
                        all_vars.append(i)
                        seen.add(i)
                    if j not in seen:
                        all_vars.append(j)
                        seen.add(j)
                self.variable_ids = all_vars
                self.variable_index = {var_id: idx for idx, var_id in enumerate(self.variable_ids)}
                n = len(self.variable_ids)
                self.linear = np.zeros(n, dtype=np.float64)
            
            self.quadratic = np.zeros((n, n), dtype=np.float64)
            for (i, j), val in quadratic.items():
                val_f64 = float(val)
                idx_i = self.variable_index[i]
                idx_j = self.variable_index[j]
                self.quadratic[idx_i, idx_j] = self.quadratic[idx_j, idx_i] = val_f64
        else:
            self.quadratic = np.asarray(quadratic, dtype=np.float64)
            if self.quadratic.shape != (n, n):
                raise ValueError(
                    f"Quadratic shape {self.quadratic.shape} != ({n},{n})"
                )
            # For array input, assume variable IDs are 0, 1, 2, ..., n-1
            if not hasattr(self, 'variable_ids'):
                self.variable_ids = list(range(n))
                self.variable_index = {var_id: idx for idx, var_id in enumerate(self.variable_ids)}

        self.offset = float(offset)
        self.vartype = Vartype(vartype)
        self.num_variables = n

        # Validate NaN and Inf
        if np.any(np.isnan(self.linear)) or np.any(np.isinf(self.linear)):
            raise ValueError("Linear coefficients cannot contain NaN or Inf")
        if np.any(np.isnan(self.quadratic)) or np.any(np.isinf(self.quadratic)):
            raise ValueError("Quadratic coefficients cannot contain NaN or Inf")
        if np.isnan(self.offset) or np.isinf(self.offset):
            raise ValueError("Offset cannot contain NaN or Inf")

        # Validate symmetry
        if n > 0 and not np.allclose(self.quadratic, self.quadratic.T, rtol=1e-14):
            raise ValueError("Quadratic matrix must be symmetric")

        # Validate coefficient bounds
        if np.any(np.abs(self.quadratic) > 1e12) or np.any(np.abs(self.linear) > 1e12):
            raise OverflowError("Coefficient magnitude exceeds 1e12")

    def energy(self, sample: Union[Dict[int, int], np.ndarray]) -> float:
        """Calculate the energy of a single sample.
        
        For BINARY variables, the energy is calculated as:
        E(x) = x^T Q x + offset = Σ_i Q_ii x_i + Σ_{i<j} Q_ij x_i x_j + offset
        
        For SPIN variables, the energy is calculated as:
        H(s) = s^T J s + h^T s + offset = Σ_i h_i s_i + Σ_{i<j} J_ij s_i s_j + offset
        
        Args:
            sample: Sample as dict {variable: value} or array.
            
        Returns:
            Energy of the sample.
            
        Raises:
            ValueError: If sample length doesn't match model.
        """
        # Handle dict input for sample
        if isinstance(sample, dict):
            # For SPIN variables, default value is -1
            sample_vec = np.full(self.num_variables, -1, dtype=np.float64)
            for var_id, val in sample.items():
                if var_id in self.variable_index:
                    idx = self.variable_index[var_id]
                    sample_vec[idx] = val
        else:
            sample_vec = np.asarray(sample).flatten()
            if len(sample_vec) != self.num_variables:
                raise ValueError("Sample length mismatch")

        if self.vartype == Vartype.BINARY:
            # For BINARY: E(x) = x^T Q x + offset
            # Since Q is stored as a full symmetric matrix, we need to apply the 0.5 factor
            # to avoid double counting the off-diagonal terms
            linear = np.dot(self.linear, sample_vec)
            quadratic = 0.5 * np.dot(sample_vec, np.dot(self.quadratic, sample_vec))
            return linear + quadratic + self.offset
        else:  # SPIN
            # For SPIN: H(s) = s^T J s + h^T s + offset
            # Since J is stored as a full symmetric matrix, we need to apply the 0.5 factor
            # to avoid double counting the off-diagonal terms
            linear = np.dot(self.linear, sample_vec)
            quadratic = 0.5 * np.dot(sample_vec, np.dot(self.quadratic, sample_vec))
            return linear + quadratic + self.offset

    def energies(self, samples: np.ndarray) -> np.ndarray:
        """Calculate energies for multiple samples.
        
        For BINARY variables, the energy is calculated as:
        E(x) = x^T Q x + offset = Σ_i Q_ii x_i + Σ_{i<j} Q_ij x_i x_j + offset
        
        For SPIN variables, the energy is calculated as:
        H(s) = s^T J s + h^T s + offset = Σ_i h_i s_i + Σ_{i<j} J_ij s_i s_j + offset
        
        Args:
            samples: Samples as 2D array (n_samples, n_variables) or list of dicts.
            
        Returns:
            Array of energies for each sample.
            
        Raises:
            ValueError: If sample dimensions don't match model.
        """
        # Handle list of dicts
        if isinstance(samples, list) and samples and isinstance(samples[0], dict):
            samples_array = np.zeros((len(samples), self.num_variables), dtype=np.float64)
            for i, sample in enumerate(samples):
                for var_id, val in sample.items():
                    if var_id in self.variable_index:
                        idx = self.variable_index[var_id]
                        samples_array[i, idx] = val
            samples = samples_array
        else:
            samples = np.asarray(samples, dtype=np.float64)
            if samples.shape[1] != self.num_variables:
                raise ValueError(
                    f"Sample dimension {samples.shape[1]} != {self.num_variables}"
                )

        if self.vartype == Vartype.BINARY:
            # For BINARY: E(x) = x^T Q x + offset
            # Since Q is stored as a full symmetric matrix, we need to apply the 0.5 factor
            # to avoid double counting the off-diagonal terms
            lin_term = np.dot(samples, self.linear)
            quad_term = 0.5 * np.sum(np.dot(samples, self.quadratic) * samples, axis=1)
            return lin_term + quad_term + self.offset
        else:  # SPIN
            # For SPIN: H(s) = s^T J s + h^T s + offset
            # Since J is stored as a full symmetric matrix, we need to apply the 0.5 factor
            # to avoid double counting the off-diagonal terms
            lin_term = np.dot(samples, self.linear)
            quad_term = 0.5 * np.sum(np.dot(samples, self.quadratic) * samples, axis=1)
            return lin_term + quad_term + self.offset

    def copy(self) -> "BinaryQuadraticModel":
        """Create a deep copy of the BQM.

        Returns:
            A new BQM with independent copies of all data.
        """
        bqm = BinaryQuadraticModel.__new__(BinaryQuadraticModel)
        bqm.linear = self.linear.copy()
        bqm.quadratic = self.quadratic.copy()
        bqm.offset = self.offset
        bqm.vartype = self.vartype
        bqm.num_variables = self.num_variables
        bqm.variable_ids = self.variable_ids.copy()
        bqm.variable_index = self.variable_index.copy()
        return bqm

    def to_dense(self) -> np.ndarray:
        """Return the dense quadratic matrix representation.

        Returns:
            Dense quadratic matrix.
        """
        return self.quadratic.copy()