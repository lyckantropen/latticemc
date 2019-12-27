import numpy as np
from numba import njit
from randomQuaternion import randomQuaternion, wiggleQuaternion
from definitions import particle, Lattice, LatticeState
    
@njit(cache=True)
def initializeRandomQuaternions(xlattice):
    for i in np.ndindex(xlattice.shape[:-1]):
        xlattice[i] = randomQuaternion(1)

def initializeRandom(lattice: Lattice):
    initializeRandomQuaternions(lattice.particles['x'])
    lattice.particles['p'] = np.random.choice([-1,1], lattice.size).reshape(lattice.shape)

def initializeOrdered(lattice: Lattice):
    lattice.particles['x'] = [1,0,0,0]
    lattice.particles['p'] = 1
    
def initializePartiallyOrdered(lattice: Lattice):
    #<parity>~0.5, "mostly" biaxial nematic
    lattice.particles['p'] = np.random.choice([-1,1], lattice.particles.size, p=[0.25,0.75]).reshape(lattice.particles.shape)
    lattice.particles['x'] = [1,0,0,0]
    for i in np.ndindex(lattice.particles.shape):
        lattice.particles['x'][i] = wiggleQuaternion(lattice.particles['x'][i], 0.02)