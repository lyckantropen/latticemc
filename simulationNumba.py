"""
Monte Carlo lattice simulation accelerated using Numba
(much, much faster than plain Python)
"""
import numpy as np
from numba import jit, njit
from randomQuaternion import randomQuaternion, wiggleQuaternion
from definitions import particle, Lattice, LatticeState
from tensorTools import dot6, dot10, T20AndT22In6Coordinates, quaternionToOrientation


@njit(cache=True)
def _getPropertiesFromOrientation(x, parity):
    """
    Calculate per-particle properties from the orientation
    quaternion 'x' and parity 'p'. Returns ex, ey, ez, t32.
    """
    ex, ey, ez = quaternionToOrientation(x)

    t32 = np.zeros(10, np.float32)

    # 000
    t32[0] = 6.0 * (ex[0] * ey[0] * ez[0])  # 1
    # 100
    t32[1] = 2.0 * (ex[0] * ey[0] * ez[1] +  # 3
                    ex[0] * ey[1] * ez[0] +
                    ex[1] * ey[0] * ez[0])
    # 110
    t32[2] = 2.0 * (ex[0] * ey[1] * ez[1] +  # 3
                    ex[1] * ey[0] * ez[1] +
                    ex[1] * ey[1] * ez[0])
    # 111
    t32[3] = 6.0 * (ex[1] * ey[1] * ez[1])  # 1
    # 200
    t32[4] = 2.0 * (ex[0] * ey[0] * ez[2] +  # 3
                    ex[0] * ey[2] * ez[0] +
                    ex[2] * ey[0] * ez[0])
    # 210
    t32[5] = (ex[0] * ey[1] * ez[2] +     # 6
              ex[0] * ey[2] * ez[1] +
              ex[1] * ey[0] * ez[2] +
              ex[2] * ey[0] * ez[1] +
              ex[1] * ey[2] * ez[0] +
              ex[2] * ey[1] * ez[0])
    # 211
    t32[6] = 2.0 * (ex[1] * ey[1] * ez[2] +  # 3
                    ex[1] * ey[2] * ez[1] +
                    ex[2] * ey[1] * ez[1])
    # 220
    t32[7] = 2.0 * (ex[2] * ey[2] * ez[0] +  # 3
                    ex[0] * ey[2] * ez[2] +
                    ex[2] * ey[0] * ez[2])
    # 221
    t32[8] = 2.0 * (ex[2] * ey[2] * ez[1] +  # 3
                    ex[1] * ey[2] * ez[2] +
                    ex[2] * ey[1] * ez[2])
    # 222
    t32[9] = 6.0 * (ex[2] * ey[2] * ez[2])  # 1

    t32 *= (parity/np.sqrt(6))

    return ex, ey, ez, t32


@njit(cache=True)
def _getNeighbors(center, lattice):
    """
    For a given 3-dimensional 'center' index and a 3d array
    'lattice', return the values of 'lattice' at the nearest
    neighbor sites, obeying periodic boundary conditions.
    """
    ind = np.mod(np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ]) + center, lattice.shape[0])
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
    Q = t20 + lam*np.sqrt(2)*t22
    energy = 0
    for i in range(nx.shape[0]):
        exi, eyi, ezi, t32i = _getPropertiesFromOrientation(nx[i], npi[i])
        t20i, t22i = T20AndT22In6Coordinates(exi, eyi, ezi)
        Qi = t20i + lam*np.sqrt(2)*t22i
        energy += (-dot6(Q, Qi)-tau*dot10(t32, t32i))/2

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
        if np.random.random() < np.exp(-dE/temperature):
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
        particle = lattice[tuple(_i)]
        nx = _getNeighbors(_i, lattice['x'])
        npi = _getNeighbors(_i, lattice['p'][..., np.newaxis])
        energy1 = _getEnergy(
            particle['x'], particle['p'], nx, npi, lam=lam, tau=tau)

        # adjust x
        x_ = wiggleQuaternion(particle['x'], wiggleRate)
        energy2 = _getEnergy(x_, particle['p'], nx, npi, lam=lam, tau=tau)
        if _metropolis(2*(energy2-energy1), temperature):
            particle['x'] = x_
            particle['energy'] = energy2

        # adjust p
        if tau != 0:
            p_ = -particle['p']
            energy2 = _getEnergy(particle['x'], p_, nx, npi, lam=lam, tau=tau)
            if _metropolis(2*(energy2-energy1), temperature):
                particle['p'] = p_
                particle['energy'] = energy2

        # instead of using the same t20 and t22 tensors, calculate depending on lambda parameter
        a, b, c, particle['t32'] = _getPropertiesFromOrientation(particle['x'], particle['p'])
        if lam < (np.sqrt(1/6)-1e-3):
            ex, ey, ez = a, b, c
        if lam > (np.sqrt(1/6)+1e-3):
            ex, ey, ez = c, a, b
        if (lam - np.sqrt(1/6)) < 1e-3:
            ex, ey, ez = b, c, a
        particle['t20'], particle['t22'] = T20AndT22In6Coordinates(ex, ey, ez)


@jit(forceobj=True, nopython=False, cache=True)
def _getLatticeAverages(lattice):
    """
    Calculate state average of per-particle properties
    as specified in the 'particle' data type.
    """
    avg = np.zeros(1, dtype=particle)
    avg['x'] = lattice['x'].mean(axis=(0, 1, 2))
    avg['t20'] = lattice['t20'].mean(axis=(0, 1, 2))
    avg['t22'] = lattice['t22'].mean(axis=(0, 1, 2))
    avg['t32'] = lattice['t32'].mean(axis=(0, 1, 2))
    avg['p'] = lattice['p'].mean()
    avg['energy'] = lattice['energy'].mean()
    return avg[0]


@jit(forceobj=True, nopython=False, cache=True)
def doLatticeStateUpdate(state: LatticeState):
    """
    Perform one update of the lattice state, updating
    particles at random (some particles can be updated
    many times, others ommited). Then, update the 
    state averages of per-particle properties.
    """
    # update particles at random
    indexes = (state.lattice.particles['index']
               .reshape(-1, 3)[np.random.randint(0,
                                                 state.lattice.particles.size,
                                                 state.lattice.particles.size)])
    _doOrientationSweep(state.lattice.particles,
                        indexes,
                        state.temperature,
                        state.lam,
                        state.tau,
                        state.wiggleRate)
    state.iterations += 1
    state.latticeAverages = np.append(state.latticeAverages,
                                      _getLatticeAverages(state.lattice.particles))
    state.wiggleRateValues = np.append(state.wiggleRateValues, state.wiggleRate)
