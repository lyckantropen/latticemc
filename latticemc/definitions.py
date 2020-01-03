import numpy as np
from dataclasses import dataclass, field
from decimal import Decimal

# per-particle degrees of freedom
particleDoF = np.dtype({
    'names': [ 'x', 'p' ],
    'formats': [ (np.float32, (4,)), np.int8 ]
}, align=True)

# declaration of the above for use in OpenCL
particleDoF_cdecl = """
typedef struct __attribute__ ((packed)) {
    float4  x;
    char p;
} particleDoF;
"""
# per-particle properties (other than DoF)
particleProps = np.dtype({
    'names': ['index',
              't20',
              't22',
              't32',
              'energy',
              'p'
              ],
    'formats': [(np.int, (3,)),
                (np.float32, (6,)),
                (np.float32, (6,)),
                (np.float32, (10,)),
                np.float32,
                np.float32
                ]    
}, align=True)

# declaration of the above for use in OpenCL
particleProps_cdecl = """
typedef struct __attribute__ ((packed)) {
    ushort3 index;
    float  t20[6];
    float  t22[6];
    float  t32[10];
    float energy;
    float p;
} particleProps;
"""

# data type to store the order parameters
gatheredOrderParameters = np.dtype([
    ('energy', float),
    ('q0', float),
    ('q2', float),
    ('w', float),
    ('p', float),
    ('d322', float)
])

# data type to store statistics
simulationStats = np.dtype([
    ('wiggleRate', float),
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
        self.particles = np.zeros((self.X, self.Y, self.Z), dtype=particleDoF)
        self.properties = np.zeros((self.X, self.Y, self.Z), dtype=particleProps)
        self.properties['index'] = np.array(
            list(np.ndindex((self.X, self.Y, self.Z)))).reshape(self.X, self.Y, self.Z, 3)


@dataclass
class DefiningParameters:
    temperature: Decimal
    tau: Decimal
    lam: Decimal

    def __hash__(self):
        return (self.temperature, self.tau, self.lam).__hash__()


@dataclass
class LatticeState:
    """
    Represents a given state of a molecular lattice at a given point in the simulation
    """
    parameters: DefiningParameters
    lattice: Lattice

    iterations: int = 0
    wiggleRate: float = 1
    latticeAverages: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=particleProps))  # instantaneous order parameters


@dataclass
class OrderParametersHistory:
    orderParameters: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=gatheredOrderParameters))
    fluctuations: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=gatheredOrderParameters))
    stats: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=simulationStats))
