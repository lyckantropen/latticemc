from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

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
])


@dataclass
class Lattice:
    """
    Represents the molecular lattice
    """
    X: int
    Y: int
    Z: int
    particles: np.ndarray = field(default=None)
    properties: np.ndarray = field(default=None)

    def __post_init__(self):
        self.particles = np.zeros((self.X, self.Y, self.Z), dtype=particle_dof)
        self.properties = np.zeros((self.X, self.Y, self.Z), dtype=particle_props)
        self.properties['index'] = np.array(
            list(np.ndindex((self.X, self.Y, self.Z)))).reshape(self.X, self.Y, self.Z, 3)


@dataclass
class DefiningParameters:
    """
    Uniquely defines the parameters of the model.
    """
    temperature: Decimal
    tau: Decimal
    lam: Decimal

    def __hash__(self):
        return (self.temperature, self.tau, self.lam).__hash__()


@dataclass
class LatticeState:
    """
    Represents the state of a molecular lattice at a given point in the simulation.
    """
    parameters: DefiningParameters
    lattice: Lattice

    iterations: int = 0
    wiggle_rate: float = 1
    lattice_averages: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=particle_props))  # instantaneous order parameters


@dataclass
class OrderParametersHistory:
    """
    Values of order parameters, fluctuation of order parameters and simulation statistics as a function of time.
    """
    order_parameters: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=gathered_order_parameters))
    fluctuations: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=gathered_order_parameters))
    stats: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=simulation_stats))
