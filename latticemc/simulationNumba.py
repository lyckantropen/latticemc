"""
Monte Carlo lattice simulation accelerated using Numba
(much, much faster than plain Python)
"""
import numpy as np
from numba import jit, njit
from .randomQuaternion import wiggleQuaternion
from .definitions import LatticeState, particleProps
from .tensorTools import dot6, dot10, T20AndT22In6Coordinates, T32In10Coordinates, quaternionToOrientation, SQRT2
from .orderParameters import biaxialOrdering


@njit(cache=True)
def _getPropertiesFromOrientation(x, parity):
    """
    Calculate per-particle properties from the orientation
    quaternion 'x' and parity 'p'. Returns ex, ey, ez, t32.
    """
    ex, ey, ez = quaternionToOrientation(x)
    t32 = T32In10Coordinates(ex, ey, ez)
    t32 *= parity
    return ex, ey, ez, t32


@njit(cache=True)
def _getNeighbors(center, lattice):
    """
    For a given 3-dimensional 'center' index and a 3d array
    'lattice', return the values of 'lattice' at the nearest
    neighbor sites, obeying periodic boundary conditions.
    """
    neighs = 2 * (lattice.ndim - 1)
    ind = np.zeros((neighs, lattice.ndim - 1), dtype=np.int64)
    for d in range(lattice.ndim - 1):
        ind[d * 2, d] = np.mod(center[d] + 1, lattice.shape[d])
        ind[d * 2 + 1, d] = np.mod(center[d] - 1, lattice.shape[d])

    n = np.zeros((6, lattice.shape[-1]), lattice.dtype)
    n[0] = lattice[ind[0, 0], ind[0, 1], ind[0, 2]]
    n[1] = lattice[ind[1, 0], ind[1, 1], ind[1, 2]]
    n[2] = lattice[ind[2, 0], ind[2, 1], ind[2, 2]]
    n[3] = lattice[ind[3, 0], ind[3, 1], ind[3, 2]]
    n[4] = lattice[ind[4, 0], ind[4, 1], ind[4, 2]]
    n[5] = lattice[ind[5, 0], ind[5, 1], ind[5, 2]]
    return n


@njit(cache=True)
def _getEnergy(x, p, nx, npi, lam, tau):
    """
    Calculate the per-particle interaction energy
    according to the Hamiltonian postulated in PRE79.
    """
    ex, ey, ez, t32 = _getPropertiesFromOrientation(x, p)
    t20, t22 = T20AndT22In6Coordinates(ex, ey, ez)
    Q = t20 + lam * SQRT2 * t22
    energy = 0
    for i in range(nx.shape[0]):
        exi, eyi, ezi, t32i = _getPropertiesFromOrientation(nx[i], npi[i])
        t20i, t22i = T20AndT22In6Coordinates(exi, eyi, ezi)
        Qi = t20i + lam * SQRT2 * t22i
        energy += (-dot6(Q, Qi) - tau * dot10(t32, t32i)) / 2

    return energy


@njit(cache=True)
def _metropolis(dE, temperature):
    """
    Decide the microstate evolution according to the
    Metropolis algorithm at given temperature.
    """
    if dE < 0:
        return True
    else:
        if np.random.random() < np.exp(-dE / temperature):
            return True
    return False


@jit(forceobj=True, nopython=False, parallel=True)
def _doOrientationSweep(lattice, indexes, temperature, lam, tau, wiggleRate):
    """
    Execute the Metropolis microstate evolution of 'lattice'
    of particles specified by 'indexes' at given temperature, according
    to the PRE79 parameters 'lam' (lamba) and 'tau'. The orientation is
    updated using a random walk with radius 'wiggleRate'.
    """
    for _i in indexes:
        particle = lattice.particles[tuple(_i)]
        props = lattice.properties[tuple(_i)]
        nx = _getNeighbors(_i, lattice.particles['x'])
        npi = _getNeighbors(_i, lattice.particles['p'][..., np.newaxis])
        energy1 = _getEnergy(particle['x'], particle['p'], nx, npi, lam=lam, tau=tau)

        # adjust x
        x_ = wiggleQuaternion(particle['x'], wiggleRate)
        energy2 = _getEnergy(x_, particle['p'], nx, npi, lam=lam, tau=tau)
        if _metropolis(2 * (energy2 - energy1), temperature):
            particle['x'] = x_
            props['energy'] = energy2

        # adjust p
        if tau != 0:
            p_ = -particle['p']
            energy2 = _getEnergy(particle['x'], p_, nx, npi, lam=lam, tau=tau)
            if _metropolis(2 * (energy2 - energy1), temperature):
                particle['p'] = p_
                props['energy'] = energy2

        # instead of using the same t20 and t22 tensors, calculate depending on lambda parameter
        a, b, c, props['t32'] = _getPropertiesFromOrientation(particle['x'], particle['p'])
        o = biaxialOrdering(lam)
        if o == 1:
            ex, ey, ez = a, b, c
        if o == -1:
            ex, ey, ez = c, a, b
        if o == 0:
            ex, ey, ez = b, c, a
        props['t20'], props['t22'] = T20AndT22In6Coordinates(ex, ey, ez)


@jit(forceobj=True, nopython=False, cache=True)
def _getLatticeAverages(lattice):
    """
    Calculate state average of per-particle properties
    as specified in the 'particle' data type.
    """
    avg = np.zeros(1, dtype=particleProps)
    avg['t20'] = lattice.properties['t20'].mean(axis=(0, 1, 2))
    avg['t22'] = lattice.properties['t22'].mean(axis=(0, 1, 2))
    avg['t32'] = lattice.properties['t32'].mean(axis=(0, 1, 2))
    # parity is calculated from the DoF, but needs to be promoted to float
    avg['p'] = lattice.particles['p'].astype(np.float32).mean()
    avg['energy'] = lattice.properties['energy'].mean()
    return avg


@jit(forceobj=True, nopython=False, cache=True)
def doLatticeStateUpdate(state: LatticeState):
    """
    Perform one update of the lattice state, updating
    particles at random (some particles can be updated
    many times, others ommited). Then, update the
    state averages of per-particle properties.
    """
    # update particles at random
    indexes = (state.lattice.properties['index']
               .reshape(-1, 3)[np.random.randint(0,
                                                 state.lattice.particles.size,
                                                 state.lattice.particles.size)])
    _doOrientationSweep(state.lattice,
                        indexes,
                        float(state.parameters.temperature),
                        float(state.parameters.lam),
                        float(state.parameters.tau),
                        state.wiggleRate)
    state.iterations += 1
    state.latticeAverages = _getLatticeAverages(state.lattice)
