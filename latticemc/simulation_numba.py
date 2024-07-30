"""Monte Carlo lattice simulation accelerated using Numba (much, much faster than plain Python)."""
from typing import Tuple

import numba as nb
import numpy as np
from nptyping import Float32, Int32, NDArray, Object, Shape

from .definitions import Lattice, LatticeState, particle_props
from .order_parameters import biaxial_ordering
from .random_quaternion import wiggle_quaternion
from .tensor_tools import SQRT2, dot6, dot10, quaternion_to_orientation, t20_and_t22_in_6_coordinates, t32_in_10_coordinates


@nb.njit([nb.types.UniTuple(nb.float32[:], 4)(nb.float32[:], nb.int8)], cache=True)
def _get_properties_from_orientation(x: NDArray[Shape['4'], Float32],
                                     parity: np.int8
                                     ) -> Tuple[NDArray[Shape['3'], Float32], NDArray[Shape['3'], Float32], NDArray[Shape['3'], Float32], NDArray[Shape['10'], Float32]]:
    """
    Calculate per-particle properties from the orientation quaternion 'x' and parity 'p'.

    Returns
    -------
    ex: NDArray[(3,), np.float32]
        x vector of particle orientation
    ey: NDArray[(3,), np.float32]
        y vector of particle orientation
    ez: NDArray[(3,), np.float32]
        z vector of particle orientation
    t32: NDArray[Shape['10'], np.float32]
        the T32 tensor at the particle
    """
    ex, ey, ez = quaternion_to_orientation(x)
    t32 = t32_in_10_coordinates(ex, ey, ez)
    t32 *= parity
    return ex, ey, ez, t32


@nb.njit(cache=True)
def _get_neighbors(center: NDArray[Shape['3'], Int32], lattice: NDArray[Shape, Object]):
    """
    Get nearest neighbors of a particle site on the lattice.

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


@nb.njit([nb.float32(nb.float32[:], nb.int8, nb.float32[:, :], nb.int8[:], nb.float32, nb.float32),
          nb.float32(nb.float32[:], nb.int64, nb.float32[:, :], nb.int64[:], nb.float32, nb.float32)],
         cache=True)
def _get_energy(x, p, nx, npi, lam, tau):
    """Calculate the per-particle interaction energy according to the Hamiltonian postulated in PRE79."""
    ex, ey, ez, t32 = _get_properties_from_orientation(x, p)
    t20, t22 = t20_and_t22_in_6_coordinates(ex, ey, ez)
    q = t20 + lam * SQRT2 * t22
    energy = 0
    for i in range(nx.shape[0]):
        exi, eyi, ezi, t32i = _get_properties_from_orientation(nx[i], npi[i].item())
        t20i, t22i = t20_and_t22_in_6_coordinates(exi, eyi, ezi)
        qi = t20i + lam * SQRT2 * t22i
        energy += (-dot6(q, qi) - tau * dot10(t32, t32i)) / 2

    return energy


@nb.njit(nb.types.boolean(nb.float32, nb.float32), cache=True)
def _metropolis(d_e: np.float32, temperature: np.float32) -> bool:
    """Accept or reject the microstate evolution according to the Metropolis algorithm at given temperature."""
    if d_e < 0:
        return True
    else:
        if np.random.random() < np.exp(-d_e / temperature):
            return True
    return False


def _do_orientation_sweep(lattice: Lattice,
                          indexes: NDArray[Shape['*,3'], Int32],
                          temperature: np.float32,
                          lam: np.float32,
                          tau: np.float32,
                          wiggle_rate: np.float32
                          ) -> None:
    """
    Execute a single Metropolis microstate evolution of a lattice.

    Given a lattice `lattice` of particles specified by 'indexes'
    at given temperature, according to the PRE79 parameters 'lam'
    (lamba) and 'tau'. The orientation is updated using a random
    walk with radius 'wiggle_rate'.
    """
    for _i in indexes:
        particle = lattice.particles[tuple(_i)]
        props = lattice.properties[tuple(_i)]
        nx = _get_neighbors(_i, lattice.particles['x'])
        npi = _get_neighbors(_i, lattice.particles['p'][..., np.newaxis]).reshape(-1)
        energy1 = _get_energy(particle['x'], particle['p'], nx, npi, lam=lam, tau=tau)

        # adjust x
        x_ = wiggle_quaternion(particle['x'], wiggle_rate)
        energy2 = _get_energy(x_, particle['p'], nx, npi, lam=lam, tau=tau)
        if _metropolis(2 * (energy2 - energy1), temperature):
            particle['x'] = x_
            props['energy'] = energy2

        # adjust p
        if tau != 0:
            p_ = -particle['p']
            energy2 = _get_energy(particle['x'], p_, nx, npi, lam=lam, tau=tau)
            if _metropolis(2 * (energy2 - energy1), temperature):
                particle['p'] = p_
                props['energy'] = energy2

        # instead of using the same t20 and t22 tensors, calculate depending on lambda parameter
        a, b, c, props['t32'] = _get_properties_from_orientation(particle['x'], particle['p'])
        o = biaxial_ordering(lam)
        if o == 1:
            ex, ey, ez = a, b, c
        elif o == -1:
            ex, ey, ez = c, a, b
        elif o == 0:
            ex, ey, ez = b, c, a
        else:
            raise Exception(f'{o} is not a recognized biaxial ordering')
        props['t20'], props['t22'] = t20_and_t22_in_6_coordinates(ex, ey, ez)


@nb.jit(forceobj=True, nopython=False, cache=True)
def _get_lattice_averages(lattice: Lattice) -> NDArray[Shape['1'], Object]:
    """Calculate state average of per-particle properties as specified in the 'particle' data type."""
    avg = np.zeros(1, dtype=particle_props)
    avg['t20'] = lattice.properties['t20'].mean(axis=(0, 1, 2))
    avg['t22'] = lattice.properties['t22'].mean(axis=(0, 1, 2))
    avg['t32'] = lattice.properties['t32'].mean(axis=(0, 1, 2))
    # parity is calculated from the DoF, but needs to be promoted to float
    avg['p'] = lattice.particles['p'].astype(np.float32).mean()
    avg['energy'] = lattice.properties['energy'].mean()
    return avg


@nb.jit(forceobj=True, nopython=False, cache=True)
def do_lattice_state_update(state: LatticeState) -> None:
    """
    Perform one update of the lattice state.

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
    _do_orientation_sweep(state.lattice,
                          indexes,
                          float(state.parameters.temperature),
                          float(state.parameters.lam),
                          float(state.parameters.tau),
                          state.wiggle_rate)
    state.iterations += 1
    state.lattice_averages = _get_lattice_averages(state.lattice)
