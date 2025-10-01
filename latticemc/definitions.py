"""Data structures and type definitions for lattice Monte Carlo simulations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
from jaxtyping import Shaped

from .statistical import fluctuation

logger = logging.getLogger(__name__)

# per-particle degrees of freedom
particle_dof = np.dtype([
    ('x', np.float32, 4),
    ('p', np.int8)
], align=True)

# per-particle properties (other than DoF)
particle_props = np.dtype([
    ('t32', np.float32, 10),
    ('t20', np.float32, 6),
    ('t22', np.float32, 6),
    ('index', np.uint16, 3),
    ('energy', np.float32),
    ('p', np.float32)
], align=True)

# data type to store the order parameters
gathered_order_parameters = np.dtype([
    ('energy', np.float32),
    ('q0', np.float32),
    ('q2', np.float32),
    ('w', np.float32),
    ('p', np.float32),
    ('d322', np.float32)
])

# data type to store statistics
simulation_stats = np.dtype([
    ('wiggle_rate', np.float32),
    ('accepted_x', np.int32),
    ('accepted_p', np.int32),
])


def extract_scalar_value(value: Any) -> Any:
    """
    Extract scalar value from numpy arrays/scalars or return the value as-is.

    This helper function handles the common pattern of extracting values from
    numpy arrays and scalars using .item(), while leaving other types unchanged.

    Parameters
    ----------
    value : Any
        The value to extract from (numpy array/scalar or regular Python type)

    Returns
    -------
    Any
        The extracted scalar value
    """
    if isinstance(value, (np.ndarray, np.number)):
        return value.item()
    return value


@dataclass
class Lattice:
    """Represents the molecular lattice."""

    X: int
    Y: int
    Z: int
    particles: Optional[Shaped[np.ndarray, "W H L"]] = field(default=None)
    properties: Optional[Shaped[np.ndarray, "W H L"]] = field(default=None)

    def __post_init__(self):
        self.particles = np.zeros((self.X, self.Y, self.Z), dtype=particle_dof)
        self.properties = np.zeros((self.X, self.Y, self.Z), dtype=particle_props)
        self.properties['index'] = np.array(
            list(np.ndindex((self.X, self.Y, self.Z)))).reshape(self.X, self.Y, self.Z, 3)

    @property
    def size(self) -> int:
        """Total number of particles in the lattice."""
        if self.particles is not None:
            return self.particles.size
        return 0

    def to_npz_dict(self) -> dict:
        """Convert lattice data to dictionary for NPZ file saving."""
        data: Dict[str, Any] = {
            'lattice_X': self.X,
            'lattice_Y': self.Y,
            'lattice_Z': self.Z
        }
        if self.particles is not None:
            data['particles'] = self.particles
        if self.properties is not None:
            data['properties'] = self.properties
        return data

    @classmethod
    def from_npz_dict(cls, data: dict) -> Lattice:
        """Load lattice data from NPZ dictionary."""
        lattice = cls(
            X=int(extract_scalar_value(data['lattice_X'])),
            Y=int(extract_scalar_value(data['lattice_Y'])),
            Z=int(extract_scalar_value(data['lattice_Z']))
        )
        if 'particles' in data:
            lattice.particles = data['particles']
        if 'properties' in data:
            lattice.properties = data['properties']
        return lattice


@dataclass
class DefiningParameters:
    """Uniquely defines the parameters of the model."""

    temperature: Decimal
    tau: Decimal
    lam: Decimal

    def __hash__(self):
        return (self.temperature, self.tau, self.lam).__hash__()

    def to_dict(self) -> dict:
        """Convert parameters to dictionary for serialization."""
        return {
            'parameters_temperature': str(self.temperature),
            'parameters_lam': str(self.lam),
            'parameters_tau': str(self.tau)
        }

    def to_npz_dict(self) -> dict:
        """Convert parameters to dictionary for NPZ file saving."""
        return self.to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> DefiningParameters:
        """Create DefiningParameters instance from dictionary data."""
        return cls(
            temperature=Decimal(str(extract_scalar_value(data['parameters_temperature']))),
            lam=Decimal(str(extract_scalar_value(data['parameters_lam']))),
            tau=Decimal(str(extract_scalar_value(data['parameters_tau'])))
        )

    @classmethod
    def from_npz_dict(cls, data: dict) -> DefiningParameters:
        """Create DefiningParameters instance from NPZ dictionary data."""
        return cls.from_dict(data)

    def tag(self) -> str:
        """Generate a tag name from DefiningParameters.

        Creates a human-readable tag name like 'T0.90_lam0.30_tau1.00' from the parameters.
        """
        temp_str = f"T{float(self.temperature):.2f}"
        lam_str = f"lam{float(self.lam):.2f}"
        tau_str = f"tau{float(self.tau):.2f}"
        return f"{temp_str}_{lam_str}_{tau_str}"


@dataclass
class LatticeState:
    """Represents the state of a molecular lattice at a given point in the simulation."""

    parameters: DefiningParameters
    lattice: Lattice

    iterations: int = 0
    wiggle_rate: float = 1
    accepted_x: int = 0
    accepted_p: int = 0
    lattice_averages: Shaped[np.ndarray, "..."] = field(default_factory=lambda: np.empty(
        0, dtype=particle_props))  # instantaneous order parameters

    def update_from(self, other: LatticeState) -> None:
        """Update this state from another state (used in parallel tempering)."""
        # parameters stay the same
        self.lattice = other.lattice
        self.iterations = other.iterations
        self.wiggle_rate = other.wiggle_rate
        self.accepted_x = other.accepted_x
        self.accepted_p = other.accepted_p
        self.lattice_averages = other.lattice_averages

    def to_dict(self) -> dict:
        """Convert state information to dictionary for JSON serialization."""
        result = {
            'simulation_info': {
                'accepted_x': int(self.accepted_x),
                'accepted_p': int(self.accepted_p),
                'lattice_size': [self.lattice.X, self.lattice.Y, self.lattice.Z]
            },
            'lattice_averages': {}
        }

        # Add lattice averages if available
        if len(self.lattice_averages) > 0:
            try:
                lattice_avg_item = self.lattice_averages[0]
                for field_name in lattice_avg_item.dtype.names:
                    field_value = lattice_avg_item[field_name]
                    # Handle scalar values
                    if isinstance(field_value, (np.ndarray, np.number)) and field_value.size == 1:
                        result['lattice_averages'][field_name] = float(extract_scalar_value(field_value))
                    # Handle array values - take first element or convert appropriately
                    elif isinstance(field_value, (np.ndarray, list, tuple)) and len(field_value) > 0:
                        if len(field_value) == 1:
                            result['lattice_averages'][field_name] = float(field_value[0])
                        else:
                            # For multi-element arrays, store as list
                            result['lattice_averages'][field_name] = [float(x) for x in field_value]
                    else:
                        result['lattice_averages'][field_name] = float(field_value)
            except Exception as e:
                logger.warning(f"Error processing lattice_averages: {e}")
                result['lattice_averages'] = {}

        return result

    def to_npz_dict(self) -> dict:
        """Convert lattice state data to dictionary for NPZ file saving."""
        data = {
            'iterations': self.iterations,
            'accepted_x': self.accepted_x,
            'accepted_p': self.accepted_p,
            'wiggle_rate': self.wiggle_rate
        }

        lattice_data = self.lattice.to_npz_dict()
        data.update(lattice_data)

        params_data = self.parameters.to_npz_dict()
        data.update(params_data)

        return data

    @classmethod
    def from_npz_dict(cls, data: dict) -> LatticeState:
        """Load lattice state data from NPZ dictionary."""
        from . import simulation_numba

        lattice = Lattice.from_npz_dict(data)
        parameters = DefiningParameters.from_npz_dict(data)
        instance = cls(lattice=lattice, parameters=parameters)

        # Load simulation state data
        instance.iterations = int(data.get('iterations', 0))
        instance.accepted_x = int(data.get('accepted_x', 0))
        instance.accepted_p = int(data.get('accepted_p', 0))
        instance.wiggle_rate = float(data.get('wiggle_rate', 1.0))

        # Recompute lattice_averages from the loaded lattice state
        instance.lattice_averages = simulation_numba._get_lattice_averages(instance.lattice)

        return instance


@dataclass
class OrderParametersHistory:
    """Values of order parameters and simulation statistics as a function of time."""

    n_particles: int
    _order_parameters: Shaped[np.ndarray, "..."] = field(default_factory=lambda: np.empty(0, dtype=gathered_order_parameters))
    _stats: Shaped[np.ndarray, "..."] = field(default_factory=lambda: np.empty(0, dtype=simulation_stats))

    # Lists for efficient appending
    order_parameters_list: List[Shaped[np.ndarray, "1"]] = field(default_factory=list, init=False)
    stats_list: List[Shaped[np.ndarray, "1"]] = field(default_factory=list, init=False)

    def clear(self) -> None:
        """Clear all stored data."""
        self._order_parameters = np.empty(0, dtype=gathered_order_parameters)
        self._stats = np.empty(0, dtype=simulation_stats)
        self.order_parameters_list.clear()
        self.stats_list.clear()

    # CAUTION: slow
    @property
    def order_parameters(self) -> np.ndarray:
        """Get order parameters array, automatically syncing from list if needed."""
        if len(self.order_parameters_list) > 0:
            self._order_parameters = np.array(self.order_parameters_list, dtype=gathered_order_parameters)
        else:
            self._order_parameters = np.empty(0, dtype=gathered_order_parameters)
        return self._order_parameters

    @order_parameters.setter
    def order_parameters(self, value: np.ndarray) -> None:
        """Set order parameters array and sync to list."""
        self._order_parameters = value
        # Clear and repopulate the list to maintain consistency
        self.order_parameters_list.clear()
        if len(value) > 0:
            for item in value:
                self.order_parameters_list.append(item)
        else:
            self.order_parameters_list.clear()

    # CAUTION: slow
    @property
    def stats(self) -> np.ndarray:
        """Get stats array, automatically syncing from list if needed."""
        if len(self.stats_list) > 0:
            self._stats = np.array(self.stats_list, dtype=simulation_stats)
        else:
            self._stats = np.empty(0, dtype=simulation_stats)
        return self._stats

    @stats.setter
    def stats(self, value: np.ndarray) -> None:
        """Set stats array and sync to list."""
        self._stats = value
        # Clear and repopulate the list to maintain consistency
        self.stats_list.clear()
        if len(value) > 0:
            for item in value:
                self.stats_list.append(item)
        else:
            self.stats_list.clear()

    def to_dict(self) -> dict:
        """Convert latest order parameters to dictionary for JSON serialization."""
        result = {
            'latest_order_parameters': {},
            'data_counts': {
                'order_parameters': len(self.order_parameters_list),
            }
        }

        # Add latest order parameters
        if len(self.order_parameters_list) > 0:
            latest_op = self.order_parameters_list[-1]
            for field_name in latest_op.dtype.fields.keys():  # type: ignore[union-attr]
                result['latest_order_parameters'][field_name] = float(latest_op[field_name])  # type: ignore[assignment]

        return result

    def save_to_npz(self, order_parameters_path: Optional[str] = None,
                    fluctuations_path: Optional[str] = None,
                    stats_path: Optional[str] = None,
                    fluctuations_from_history: bool = False) -> None:
        """Save order parameters and stats to NPZ files. Generate fluctuations from history if requested."""
        import numpy as np

        # Get current arrays from lists or fallback to existing arrays
        order_params_array = self._get_order_parameters_array()
        stats_array = self._get_stats_array()

        if order_parameters_path and len(order_params_array) > 0:
            np.savez_compressed(order_parameters_path, order_parameters=order_params_array)

        # Generate fluctuations from order parameter history if requested
        if fluctuations_path and fluctuations_from_history and len(order_params_array) > 0:
            fluctuations_array = self.calculate_decorrelated_fluctuations()
            if len(fluctuations_array) > 0:
                np.savez_compressed(fluctuations_path, fluctuations=fluctuations_array)

        if stats_path and len(stats_array) > 0:
            np.savez_compressed(stats_path, stats=stats_array)

    def load_from_npz(self, order_parameters_path: Optional[str] = None,
                      stats_path: Optional[str] = None) -> None:
        """Load order parameters and stats from NPZ files."""
        import pathlib

        import numpy as np

        if order_parameters_path:
            op_path = pathlib.Path(order_parameters_path)
            if op_path.exists():
                op_data = np.load(op_path)
                self.order_parameters = op_data['order_parameters']

        if stats_path:
            stats_path_obj = pathlib.Path(stats_path)
            if stats_path_obj.exists():
                stats_data = np.load(stats_path_obj)
                self.stats = stats_data['stats']

    def append_order_parameters(self, item: np.ndarray) -> None:
        """Efficiently append order parameters using internal list."""
        # Handle structured arrays properly - preserve structured array format for field access
        if item.ndim == 0:  # scalar structured array
            self.order_parameters_list.append(item)
        else:  # 1D structured array with one or more elements
            for element in item:
                self.order_parameters_list.append(element)

    def append_stats(self, item: np.ndarray) -> None:
        """Efficiently append stats using internal list."""
        # Handle structured arrays properly - preserve structured array format for field access
        if item.ndim == 0:  # scalar structured array
            self.stats_list.append(item)
        else:  # 1D structured array with one or more elements
            for element in item:
                self.stats_list.append(element)

    def _get_order_parameters_array(self) -> np.ndarray:
        """Convert internal list to numpy array when needed, falling back to existing array."""
        if len(self.order_parameters_list) > 0:
            return np.array(self.order_parameters_list, dtype=gathered_order_parameters)
        # Fall back to existing array if list is empty
        return self._order_parameters

    def _get_stats_array(self) -> np.ndarray:
        """Convert internal list to numpy array when needed, falling back to existing array."""
        if len(self.stats_list) > 0:
            return np.array(self.stats_list, dtype=simulation_stats)
        # Fall back to existing array if list is empty
        return self._stats

    def calculate_decorrelated_averages(self,
                                        limit_history: Optional[int] = None,
                                        decorrelation_interval: int = 10
                                        ) -> Shaped[np.ndarray, "1"]:
        """Calculate decorrelated averages of order parameters."""
        assert gathered_order_parameters.fields is not None

        # Get current arrays from lists
        order_params_array = self._get_order_parameters_array()

        decorrelated_avgs_op = np.zeros(1, dtype=gathered_order_parameters)

        # Process order parameters if available
        if len(order_params_array) > 0:
            window_op = limit_history if limit_history is not None else len(order_params_array)
            window_op = min(window_op, len(order_params_array))  # Don't exceed available data

            if window_op > 0:
                decorrelated_op = order_params_array[-window_op::decorrelation_interval]
                for name in gathered_order_parameters.fields.keys():  # type: ignore[union-attr]
                    if name in decorrelated_op.dtype.fields.keys():  # type: ignore[union-attr]
                        decorrelated_avgs_op[name] = np.mean(decorrelated_op[name])

        return decorrelated_avgs_op

    def calculate_decorrelated_fluctuations(self,
                                            limit_history: Optional[int] = None,
                                            decorrelation_interval: int = 10
                                            ) -> Shaped[np.ndarray, "1"]:
        """Calculate decorrelated fluctuation of order parameters directly from order parameter history."""
        assert gathered_order_parameters.fields is not None

        # Get current arrays from lists
        order_params_array = self._get_order_parameters_array()

        decorrelated_fluctuation = np.zeros(1, dtype=gathered_order_parameters)

        # Process order parameters if available
        if len(order_params_array) > 0:
            window_op = limit_history if limit_history is not None else len(order_params_array)
            window_op = min(window_op, len(order_params_array))  # Don't exceed available data

            if window_op > 0:
                decorrelated_op = order_params_array[-window_op::decorrelation_interval]
                for name in gathered_order_parameters.fields.keys():  # type: ignore[union-attr]
                    if name in decorrelated_op.dtype.fields.keys():  # type: ignore[union-attr]
                        fluct = fluctuation(decorrelated_op[name]) * self.n_particles
                        decorrelated_fluctuation[name] = fluct

        return decorrelated_fluctuation
