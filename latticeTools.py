import numpy as np
from numba import njit
from randomQuaternion import randomQuaternion, wiggleQuaternion
from definitions import particle, Lattice, LatticeState
from typing import Optional
    
@njit(cache=True)
def initializeRandomQuaternions(xlattice):
    """
    Given a lattice where the size of the last dimension must
    be equal to 4, populate it with random normalized quaternions.
    """
    for i in np.ndindex(xlattice.shape[:-1]):
        xlattice[i] = randomQuaternion(1)

def initializeRandom(lattice: Lattice):
    """
    Initialize the lattice from nonbiased uniform
    distributions for orientation and parity.
    """
    initializeRandomQuaternions(lattice.particles['x'])
    lattice.particles['p'] = np.random.choice([-1,1], lattice.particles.size).reshape(lattice.particles.shape)

def initializeOrdered(lattice: Lattice, x: Optional[np.ndarray] = None, p: Optional[int] = None):
    """
    Initialize the lattice to the same value. Optionally,
    starting values for the orientation and parity
    can be specified.
    """
    x = x if x is not None else [1,0,0,0]
    p = p if p is not None else 1
    lattice.particles['x'] = x
    lattice.particles['p'] = p

def initializePartiallyOrdered(lattice: Lattice, x: Optional[np.ndarray] = None, p: Optional[int] = None):
    """
    Initialize the lattice with partial ordering. Given
    initial orientation, perturb the orientations using random
    4-vectors of radius 0.02. Given the initial parity, initialize
    25% of particles with random parity.
    
    Optionally, starting values for the orientation and parity
    can be specified.
    """
    x = x if x is not None else [1,0,0,0]
    p = p if p is not None else 1
    lattice.particles['p'] = np.random.choice([-p,p], lattice.particles.size, p=[0.25,0.75]).reshape(lattice.particles.shape)
    lattice.particles['x'] = x
    for i in np.ndindex(lattice.particles.shape):
        lattice.particles['x'][i] = wiggleQuaternion(lattice.particles['x'][i], 0.02)
