"""Data structures and type definitions for lattice Monte Carlo simulations."""

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional

import numpy as np
from jaxtyping import Shaped

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

    order_parameters: Shaped[np.ndarray, "..."] = field(default_factory=lambda: np.empty(0, dtype=gathered_order_parameters))
    fluctuations: Shaped[np.ndarray, "..."] = field(default_factory=lambda: np.empty(0, dtype=gathered_order_parameters))
    stats: Shaped[np.ndarray, "..."] = field(default_factory=lambda: np.empty(0, dtype=simulation_stats))

    def to_dict(self) -> dict:
        """Convert latest order parameters and fluctuations to dictionary for JSON serialization."""
        result = {
            'latest_order_parameters': {},
            'latest_fluctuations': {},
            'data_counts': {
                'order_parameters': len(self.order_parameters),
                'fluctuations': len(self.fluctuations)
            }
        }

        # Add latest order parameters
        if len(self.order_parameters) > 0:
            latest_op = self.order_parameters[-1]
            if self.order_parameters.dtype.names:
                for field_name in self.order_parameters.dtype.names:
                    result['latest_order_parameters'][field_name] = float(latest_op[field_name])  # type: ignore[assignment]

        # Add latest fluctuations
        if len(self.fluctuations) > 0:
            latest_fluct = self.fluctuations[-1]
            if self.fluctuations.dtype.names:
                for field_name in self.fluctuations.dtype.names:
                    result['latest_fluctuations'][field_name] = float(latest_fluct[field_name])  # type: ignore[assignment]

        return result

    def save_to_npz(self, order_parameters_path: Optional[str] = None,
                    fluctuations_path: Optional[str] = None) -> None:
        """Save order parameters and fluctuations to NPZ files."""
        import numpy as np

        if order_parameters_path and len(self.order_parameters) > 0:
            np.savez_compressed(order_parameters_path, order_parameters=self.order_parameters)

        if fluctuations_path and len(self.fluctuations) > 0:
            np.savez_compressed(fluctuations_path, fluctuations=self.fluctuations)

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
