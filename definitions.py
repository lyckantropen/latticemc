import numpy as np
from dataclasses import dataclass, field

particle = np.dtype({
    'names':   [ 'index',
                 'x',
                 't20',
                 't22',
                 't32',
                 'p',
                 'energy',],
    'formats': [ (np.int, (3,)),
                 (np.float32, (4,)),
                 (np.float32, (6,)),
                 (np.float32, (6,)),
                 (np.float32, (10,)),
                 np.float32,
                 np.float32
               ]
}, align=True)

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
    X: int
    Y: int
    Z: int
    particles: np.ndarray = field(default=None)

    def __post_init__(self):
        self.particles = np.zeros((self.Z, self.Y, self.X), dtype=particle)
        self.particles['index'] = np.array(list(np.ndindex((self.Z, self.Y, self.X)))).reshape(self.Z, self.Y, self.X,3)

@dataclass
class LatticeState:
    temperature: float
    tau: float
    lam: float
    lattice: Lattice

    iterations: int = 0
    wiggleRate: float = 1
    latticeAverages: np.ndarray = field(default_factory=lambda: np.empty(1, dtype=particle))
    wiggleRateValues: np.ndarray = field(default_factory=lambda: np.empty(1, dtype=np.float32))

