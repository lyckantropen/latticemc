import numba as nb
import numpy as np

from latticemc.definitions import Lattice, particle_dof
from latticemc.simulation_numba import _do_orientation_sweep, _get_energy, _get_neighbors
from latticemc.tensor_tools import quaternion_to_orientation, t20_t22_matrix, t32_matrix


@nb.njit
def _seed(seed):
    np.random.seed(seed)


@nb.njit
def _choice(options, count):
    return np.random.choice(options, count)


def test_get_neighbors_3x3():
    lattice = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    indices = np.array(list(np.ndindex(lattice.shape)), dtype=np.float32).reshape(3, 3, 3, 3)
    n = _get_neighbors([0, 0, 0], indices)
    assert np.all(n == [
        [1, 0, 0],
        [2, 0, 0],
        [0, 1, 0],
        [0, 2, 0],
        [0, 0, 1],
        [0, 0, 2],
    ])


def test_get_neighbors_3x6x9():
    lattice = np.arange(3 * 6 * 9).reshape(3, 6, 9)
    indices = np.array(list(np.ndindex(lattice.shape))).reshape(3, 6, 9, 3)
    n = _get_neighbors([0, 0, 0], indices)
    assert np.all(n == [
        [1, 0, 0],
        [2, 0, 0],
        [0, 1, 0],
        [0, 5, 0],
        [0, 0, 1],
        [0, 0, 8],
    ])


def test_get_energy():
    x1 = np.array([0.25091976, -0.90142864, -0.32465923, -0.13806562], dtype=np.float32)
    x2 = np.array([1, 0, 0, 0], dtype=np.float32)
    p1 = 1
    p2 = -1

    energy = _get_energy(x1, p1, np.array([x2], dtype=np.float32), np.array([p2], dtype=np.int8), 0.3, 1)

    ex1, ey1, ez1 = quaternion_to_orientation(x1)
    ex2, ey2, ez2 = quaternion_to_orientation(x2)

    t201, t221 = t20_t22_matrix(ex1, ey1, ez1)
    t202, t222 = t20_t22_matrix(ex2, ey2, ez2)
    t321 = t32_matrix(ex1, ey1, ez1) * p1
    t322 = t32_matrix(ex2, ey2, ez2) * p2

    energy_t = -1 / 2 * (np.tensordot(t201 + np.sqrt(2) * 0.3 * t221, t202 + np.sqrt(2) * 0.3 * t222) + 1 * np.tensordot(t321, t322, axes=3))
    assert np.isclose(energy, energy_t)


def test_do_orientation_sweep():
    _seed(42)
    lattice = np.zeros([3, 3, 3], dtype=particle_dof)
    lattice['x'] = np.array([1, 0, 0, 0], dtype=np.float32)
    lattice['p'] = _choice(np.array([-1, 1]), 3 * 3 * 3).reshape(lattice.shape).astype(np.int8)
    lattice_after = lattice.copy()
    indexes = np.array(list(np.ndindex((3, 3, 3)))).reshape(3 * 3 * 3, 3)

    lat = Lattice(3, 3, 3)
    lat.particles = lattice_after
    _do_orientation_sweep(lat, indexes, np.float32(2.4), np.float32(0.3), np.float32(1.0), np.float32(0.1))
    assert np.sum(lattice['x'] == lattice_after['x']) < lattice['x'].size
    assert np.sum(lattice['p'] == lattice_after['p']) < lattice['p'].size
