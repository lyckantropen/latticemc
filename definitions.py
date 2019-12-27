import numpy as np
from dataclasses import dataclass, field

# per-particle properties data type
particle = np.dtype({
    'names':   ['index',
                'x',
                't20',
                't22',
                't32',
                'p',
                'energy'
                ],
    'formats': [(np.int, (3,)),
                (np.float32, (4,)),
                (np.float32, (6,)),
                (np.float32, (6,)),
                (np.float32, (10,)),
                np.float32,
                np.float32
                ]
}, align=True)

# data type to store the order parameters
gatheredOrderParameters = np.dtype([
    ('energy', float),
    ('q0', float),
    ('q2', float),
    ('w', float),
    ('p', float),
    ('d322', float)
])

# declaration of the above for use in OpenCL
particle_cdecl = """
typedef struct __attribute__ ((packed)) {
    ushort3 index;
    float4  x;
    float  t20[6];
    float  t22[6];
    float  t32[10];
    float p;
    float energy;
} particle;
"""


@dataclass
class Lattice:
    """
    Represents the molecular lattice
    """
    X: int
    Y: int
    Z: int
    particles: np.ndarray = field(default=None)

    def __post_init__(self):
        self.particles = np.zeros((self.X, self.Y, self.Z), dtype=particle)
        self.particles['index'] = np.array(
            list(np.ndindex((self.X, self.Y, self.Z)))).reshape(self.X, self.Y, self.Z, 3)


@dataclass
class LatticeState:
    """
    Represents a given state of a molecular lattice at a given point in the simulation
    """
    temperature: float
    tau: float
    lam: float
    lattice: Lattice

    iterations: int = 0
    wiggleRate: float = 1
    latticeAverages: np.ndarray = field(default_factory=lambda: np.empty(1, dtype=particle))
    wiggleRateValues: np.ndarray = field(default_factory=lambda: np.empty(1, dtype=np.float32))

@dataclass
class OrderParametersHistory:
    orderParameters: np.ndarray = field(default_factory=lambda: np.empty(1, dtype=gatheredOrderParameters))
    fluctuations: np.ndarray = field(default_factory=lambda: np.empty(1, dtype=gatheredOrderParameters))
