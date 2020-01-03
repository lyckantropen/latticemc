from . import disableNumba  # noqa: F401
import numpy as np
from latticemc.simulationNumba import (
    _getNeighbors,
    _getEnergy,
    _doOrientationSweep
)
from latticemc.tensorTools import quaternionToOrientation
from latticemc.definitions import particleDoF, Lattice


def _t20t22Matrix(ex, ey, ez):
    xx = np.outer(ex, ex)
    yy = np.outer(ey, ey)
    zz = np.outer(ez, ez)
    t20 = np.sqrt(3 / 2) * (zz - 1 / 3 * np.eye(3))
    t22 = np.sqrt(1 / 2) * (xx - yy)
    return t20, t22


def _t32Matrix(ex, ey, ez):
    return np.sqrt(1 / 6) * (
        np.outer(np.outer(ex, ey), ez) +
        np.outer(np.outer(ez, ex), ey) +
        np.outer(np.outer(ey, ez), ex) +
        np.outer(np.outer(ex, ez), ey) +
        np.outer(np.outer(ey, ex), ez) +
        np.outer(np.outer(ez, ey), ex)
    ).reshape(3, 3, 3)


def test__getNeighbors_3x3():
    lattice = np.arange(27).reshape(3, 3, 3)
    indices = np.array(list(np.ndindex(lattice.shape))).reshape(3, 3, 3, 3)
    n = _getNeighbors([0, 0, 0], indices)
    assert np.all(n == [
        [1, 0, 0],
        [2, 0, 0],
        [0, 1, 0],
        [0, 2, 0],
        [0, 0, 1],
        [0, 0, 2],
    ])


def test__getNeighbors_3x6x9():
    lattice = np.arange(3 * 6 * 9).reshape(3, 6, 9)
    indices = np.array(list(np.ndindex(lattice.shape))).reshape(3, 6, 9, 3)
    n = _getNeighbors([0, 0, 0], indices)
    assert np.all(n == [
        [1, 0, 0],
        [2, 0, 0],
        [0, 1, 0],
        [0, 5, 0],
        [0, 0, 1],
        [0, 0, 8],
    ])


def test__getEnergy():
    x1 = np.array([0.25091976, -0.90142864, -0.32465923, -0.13806562], dtype=np.float32)
    x2 = np.array([1, 0, 0, 0], dtype=np.float32)
    p1 = 1
    p2 = -1

    energy = _getEnergy(x1, p1, np.array([x2]), np.array([p2]), 0.3, 1)

    ex1, ey1, ez1 = quaternionToOrientation(x1)
    ex2, ey2, ez2 = quaternionToOrientation(x2)

    t201, t221 = _t20t22Matrix(ex1, ey1, ez1)
    t202, t222 = _t20t22Matrix(ex2, ey2, ez2)
    t321 = _t32Matrix(ex1, ey1, ez1) * p1
    t322 = _t32Matrix(ex2, ey2, ez2) * p2

    energy_t = -1 / 2 * (np.tensordot(t201 + np.sqrt(2) * 0.3 * t221, t202 + np.sqrt(2) * 0.3 * t222) + 1 * np.tensordot(t321, t322, axes=3))
    assert np.isclose(energy, energy_t)


def test__doOrientationSweep():
    np.random.seed(42)
    lattice = np.zeros([3, 3, 3], dtype=particleDoF)
    lattice['x'] = [1, 0, 0, 0]
    lattice['p'] = np.random.choice([-1, 1], 3 * 3 * 3).reshape(lattice.shape)
    lattice_after = lattice.copy()
    indexes = np.array(list(np.ndindex((3, 3, 3)))).reshape(3 * 3 * 3, 3)

    lat = Lattice(3, 3, 3)
    lat.particles = lattice_after
    _doOrientationSweep(lat, indexes, 2.4, 0.3, 1, 0.1)
    assert np.sum(lattice['x'] == lattice_after['x']) < lattice['x'].size
    assert np.sum(lattice['p'] == lattice_after['p']) < lattice['p'].size
