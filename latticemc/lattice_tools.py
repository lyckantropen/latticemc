from typing import Optional

import numpy as np
from nptyping import Float32, NDArray, Shape
from numba import njit

from .definitions import Lattice
from .random_quaternion import random_quaternion, wiggle_quaternion


@njit(cache=True)
def initialize_random_quaternions(xlattice):
    """Given a lattice where the size of the last dimension must be equal to 4, populate it with random normalized quaternions."""
    for i in np.ndindex(xlattice.shape[:-1]):
        xlattice[i] = random_quaternion(1)


def initialize_random(lattice: Lattice):
    """Initialize the lattice from nonbiased uniform distributions for orientation and parity."""
    initialize_random_quaternions(lattice.particles['x'])
    lattice.particles['p'] = np.random.choice([-1, 1], lattice.particles.size).reshape(lattice.particles.shape)


def initialize_ordered(lattice: Lattice, x: Optional[NDArray[Shape['4'], Float32]] = None, p: Optional[int] = None):
    """
    Initialize the lattice to the same value.

    Optionally, starting values for the orientation and parity can be specified.
    """
    x = x if x is not None else [1, 0, 0, 0]
    p = p if p is not None else 1
    lattice.particles['x'] = x
    lattice.particles['p'] = p


def initialize_partially_ordered(lattice: Lattice, x: Optional[NDArray[Shape['4'], Float32]] = None, p: Optional[int] = None):
    """
    Initialize the lattice with partial ordering.

    Given initial orientation, perturb the orientations using random
    4-vectors of radius 0.02. Given the initial parity, initialize
    25% of particles with random parity. Optionally, starting values
    for the orientation and parity can be specified.
    """
    x = x if x is not None else [1, 0, 0, 0]
    p = p if p is not None else 1
    lattice.particles['p'] = np.random.choice([-p, p], lattice.particles.size, p=[0.25, 0.75]).reshape(lattice.particles.shape)
    lattice.particles['x'] = x
    for i in np.ndindex(lattice.particles.shape):
        lattice.particles['x'][i] = wiggle_quaternion(lattice.particles['x'][i], 0.02)
