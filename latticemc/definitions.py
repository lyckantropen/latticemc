"""Data structures and type definitions for lattice Monte Carlo simulations."""

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from jaxtyping import Shaped

from .statistical import fluctuation

logger = logging.getLogger(__name__)

# per-particle degrees of freedom
particle_dof = np.dtype([
    ('x', np.float32, 4),
    ('p', np.int8)
], align=True)

# declaration of the above for use in OpenCL
particle_dof_cdecl = """
typedef struct __attribute__ ((packed)) {
    float4  x;
    char p;
} particle_dof;
"""

# per-particle properties (other than DoF)
particle_props = np.dtype([
    ('t32', np.float32, 10),
    ('t20', np.float32, 6),
    ('t22', np.float32, 6),
    ('index', np.uint16, 3),
    ('energy', np.float32),
    ('p', np.float32)
], align=True)

# declaration of the above for use in OpenCL
particle_props_cdecl = """
typedef struct __attribute__ ((packed)) {
    float  t32[10];
    float  t20[6];
    float  t22[6];
    ushort3 index;
    float energy;
    float p;
} particle_props;
"""

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

    def from_npz_dict(self, data: dict) -> None:
        """Load lattice data from NPZ dictionary."""
        self.X = int(data['lattice_X'].item()) if hasattr(data['lattice_X'], 'item') else int(data['lattice_X'])
        self.Y = int(data['lattice_Y'].item()) if hasattr(data['lattice_Y'], 'item') else int(data['lattice_Y'])
        self.Z = int(data['lattice_Z'].item()) if hasattr(data['lattice_Z'], 'item') else int(data['lattice_Z'])
        if 'particles' in data:
            self.particles = data['particles']
        if 'properties' in data:
            self.properties = data['properties']


@dataclass
class DefiningParameters:
    """Uniquely defines the parameters of the model."""

    temperature: Decimal
    tau: Decimal
    lam: Decimal

    def __hash__(self):
        return (self.temperature, self.tau, self.lam).__hash__()

    def to_dict(self) -> dict:
        """Convert parameters to dictionary for JSON serialization."""
        return {
            'temperature': float(self.temperature),
            'lam': float(self.lam),
            'tau': float(self.tau)
        }

    def to_npz_dict(self) -> dict:
        """Convert parameters to dictionary for NPZ file saving."""
        return {
            'parameters_temperature': float(self.temperature),
            'parameters_lam': float(self.lam),
            'parameters_tau': float(self.tau)
        }

    def get_folder_name(self) -> str:
        """Generate a folder name from DefiningParameters.

        Creates a human-readable folder name like 'T0.90_lam0.30_tau1.00' from the parameters.
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

    def update_from(self, other: "LatticeState") -> None:
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
                    if hasattr(field_value, 'item') and field_value.size == 1:
                        result['lattice_averages'][field_name] = float(field_value.item())
                    # Handle array values - take first element or convert appropriately
                    elif hasattr(field_value, '__len__') and len(field_value) > 0:
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

    def to_npz_dict(self, include_lattice: bool = True, include_parameters: bool = True) -> dict:
        """Convert lattice state data to dictionary for NPZ file saving."""
        data = {
            'iterations': self.iterations,
            'accepted_x': self.accepted_x,
            'accepted_p': self.accepted_p
        }

        # Add lattice averages (structured array fields)
        if len(self.lattice_averages) > 0:
            lattice_avg = self.lattice_averages[0]  # Get the first (and only) element
            for field_name in lattice_avg.dtype.names:
                data[f'lattice_avg_{field_name}'] = lattice_avg[field_name]

        # Include lattice data if requested
        if include_lattice:
            lattice_data = self.lattice.to_npz_dict()
            data.update(lattice_data)

        # Include parameters data if requested
        if include_parameters:
            params_data = self.parameters.to_npz_dict()
            data.update(params_data)

        return data

    def from_npz_dict(self, data: dict, load_lattice: bool = True, load_parameters: bool = False) -> None:
        """Load lattice state data from NPZ dictionary."""
        self.iterations = int(data.get('iterations', 0))
        self.accepted_x = int(data.get('accepted_x', 0))
        self.accepted_p = int(data.get('accepted_p', 0))

        # Load lattice averages if present
        lattice_avg_fields = [key for key in data.keys() if key.startswith('lattice_avg_')]
        if lattice_avg_fields:
            # Create structured array from lattice average fields
            # This would need more complex logic to reconstruct the proper dtype
            # For now, we'll preserve the existing lattice_averages
            pass

        # Load lattice data if requested
        if load_lattice:
            self.lattice.from_npz_dict(data)


@dataclass
class OrderParametersHistory:
    """Values of order parameters, fluctuation of order parameters and simulation statistics as a function of time."""

    n_particles: int
    _order_parameters: Shaped[np.ndarray, "..."] = field(default_factory=lambda: np.empty(0, dtype=gathered_order_parameters))
    _fluctuations: Shaped[np.ndarray, "..."] = field(default_factory=lambda: np.empty(0, dtype=gathered_order_parameters))
    _stats: Shaped[np.ndarray, "..."] = field(default_factory=lambda: np.empty(0, dtype=simulation_stats))

    # Lists for efficient appending
    order_parameters_list: List[Shaped[np.ndarray, "1"]] = field(default_factory=list, init=False)
    fluctuations_list: List[Shaped[np.ndarray, "1"]] = field(default_factory=list, init=False)
    stats_list: List[Shaped[np.ndarray, "1"]] = field(default_factory=list, init=False)

    # CAUTION: slow

    @property
    def order_parameters(self) -> np.ndarray:
        """Get order parameters array, automatically syncing from list if needed."""
        if len(self.order_parameters_list) > self._order_parameters.size:
            self._order_parameters = np.array(self.order_parameters_list, dtype=gathered_order_parameters)
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

    # CAUTION: slow
    @property
    def fluctuations(self) -> np.ndarray:
        """Get fluctuations array, automatically syncing from list if needed."""
        if len(self.fluctuations_list) > self._fluctuations.size:
            self._fluctuations = np.array(self.fluctuations_list, dtype=gathered_order_parameters)
        return self._fluctuations

    @fluctuations.setter
    def fluctuations(self, value: np.ndarray) -> None:
        """Set fluctuations array and sync to list."""
        self._fluctuations = value
        # Clear and repopulate the list to maintain consistency
        self.fluctuations_list.clear()
        if len(value) > 0:
            for item in value:
                self.fluctuations_list.append(item)

    # CAUTION: slow
    @property
    def stats(self) -> np.ndarray:
        """Get stats array, automatically syncing from list if needed."""
        if len(self.stats_list) > self._stats.size:
            self._stats = np.array(self.stats_list, dtype=simulation_stats)
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

    def to_dict(self) -> dict:
        """Convert latest order parameters and fluctuations to dictionary for JSON serialization."""
        result = {
            'latest_order_parameters': {},
            'latest_fluctuations': {},
            'data_counts': {
                'order_parameters': len(self.order_parameters_list),
                'fluctuations': len(self.fluctuations_list)
            }
        }

        # Add latest order parameters
        if len(self.order_parameters_list) > 0:
            latest_op = self.order_parameters_list[-1]
            for field_name in latest_op.dtype.fields.keys():  # type: ignore[union-attr]
                result['latest_order_parameters'][field_name] = float(latest_op[field_name])  # type: ignore[assignment]

        # Add latest fluctuations
        if len(self.fluctuations_list) > 0:
            latest_fluct = self.fluctuations_list[-1]
            for field_name in latest_fluct.dtype.fields.keys():  # type: ignore[union-attr]
                result['latest_fluctuations'][field_name] = float(latest_fluct[field_name])  # type: ignore[assignment]

        return result

    def save_to_npz(self, order_parameters_path: Optional[str] = None,
                    fluctuations_path: Optional[str] = None,
                    fluctuations_from_history: bool = False) -> None:
        """Save order parameters and fluctuations to NPZ files."""
        import numpy as np

        # Get current arrays from lists or fallback to existing arrays
        order_params_array = self._get_order_parameters_array()
        if fluctuations_from_history:
            fluctuations_array = self.calculate_decorrelated_fluctuations()
        else:
            fluctuations_array = self._get_fluctuations_array()

        if order_parameters_path and len(order_params_array) > 0:
            np.savez_compressed(order_parameters_path, order_parameters=order_params_array)

        if fluctuations_path and len(fluctuations_array) > 0:
            np.savez_compressed(fluctuations_path, fluctuations=fluctuations_array)

    def load_from_npz(self, order_parameters_path: Optional[str] = None,
                      fluctuations_path: Optional[str] = None) -> None:
        """Load order parameters and fluctuations from NPZ files."""
        import pathlib

        import numpy as np

        if order_parameters_path:
            op_path = pathlib.Path(order_parameters_path)
            if op_path.exists():
                op_data = np.load(op_path)
                self.order_parameters = op_data['order_parameters']

        if fluctuations_path:
            fluc_path = pathlib.Path(fluctuations_path)
            if fluc_path.exists():
                fluc_data = np.load(fluc_path)
                self.fluctuations = fluc_data['fluctuations']

    def append_order_parameters(self, item: np.ndarray) -> None:
        """Efficiently append order parameters using internal list."""
        # Handle structured arrays properly - preserve structured array format for field access
        if item.ndim == 0:  # scalar structured array
            self.order_parameters_list.append(item)
        else:  # 1D structured array with one or more elements
            for element in item:
                self.order_parameters_list.append(element)

    def append_fluctuations(self, item: np.ndarray) -> None:
        """Efficiently append fluctuations using internal list."""
        # Handle structured arrays properly - preserve structured array format for field access
        if item.ndim == 0:  # scalar structured array
            self.fluctuations_list.append(item)
        else:  # 1D structured array with one or more elements
            for element in item:
                self.fluctuations_list.append(element)

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

    def _get_fluctuations_array(self) -> np.ndarray:
        """Convert internal list to numpy array when needed, falling back to existing array."""
        if len(self.fluctuations_list) > 0:
            return np.array(self.fluctuations_list, dtype=gathered_order_parameters)
        # Fall back to existing array if list is empty
        return self._fluctuations

    def _get_stats_array(self) -> np.ndarray:
        """Convert internal list to numpy array when needed, falling back to existing array."""
        if len(self.stats_list) > 0:
            return np.array(self.stats_list, dtype=simulation_stats)
        # Fall back to existing array if list is empty
        return self._stats

    def calculate_decorrelated_averages(self,
                                        limit_history: Optional[int] = None,
                                        decorrelation_interval: int = 10
                                        ) -> Tuple[Shaped[np.ndarray, "1"], Shaped[np.ndarray, "1"]]:
        """Calculate decorrelated averages of order parameters."""
        assert gathered_order_parameters.fields is not None

        # Get current arrays from lists
        order_params_array = self._get_order_parameters_array()
        fluctuations_array = self._get_fluctuations_array()

        decorrelated_avgs_op = np.zeros(1, dtype=gathered_order_parameters)
        decorrelated_avgs_fl = np.zeros(1, dtype=gathered_order_parameters)

        # Process order parameters if available
        if len(order_params_array) > 0:
            window_op = limit_history if limit_history is not None else len(order_params_array)
            window_op = min(window_op, len(order_params_array))  # Don't exceed available data

            if window_op > 0:
                decorrelated_op = order_params_array[-window_op::decorrelation_interval]
                for name in gathered_order_parameters.fields.keys():  # type: ignore[union-attr]
                    if name in decorrelated_op.dtype.fields.keys():  # type: ignore[union-attr]
                        decorrelated_avgs_op[name] = np.mean(decorrelated_op[name])

        # Process fluctuations if available
        if len(fluctuations_array) > 0:
            window_fl = limit_history if limit_history is not None else len(fluctuations_array)
            window_fl = min(window_fl, len(fluctuations_array))  # Don't exceed available data

            if window_fl > 0:
                # fluctuations are already decorrelated, just take the mean over the window
                for name in gathered_order_parameters.fields.keys():  # type: ignore[union-attr]
                    if name in fluctuations_array.dtype.fields.keys():  # type: ignore[union-attr]
                        decorrelated_avgs_fl[name] = np.mean(fluctuations_array[name])

        return decorrelated_avgs_op, decorrelated_avgs_fl

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
