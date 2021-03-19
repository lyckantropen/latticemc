import numba as nb
import numpy as np

from latticemc.random_quaternion import random_quaternion, wiggle_quaternion


@nb.njit
def _seed(seed):
    np.random.seed(seed)


def test_random_quaternion_norm():
    _seed(42)
    x = random_quaternion(1.4)
    assert np.isclose(np.linalg.norm(x), 1.4)


def test_wiggle_quaternion_norm():
    _seed(42)
    x = np.array([1, 0, 0, 0], dtype=np.float32)
    y = wiggle_quaternion(x, 0.1)
    assert np.linalg.norm(x - y) < 0.1


def test_random_quaternion_uniform():
    _seed(42)
    x = np.zeros((10000, 4), dtype=np.float32)
    for i in range(10000):
        x[i, :] = random_quaternion(1)
    avgx = x.mean(axis=0)
    mx = np.linalg.norm(avgx)
    assert np.isclose(mx, 0, rtol=0.015, atol=0.015)


def test_wiggle_quaternion_uniform():
    _seed(42)
    x = np.array([1, 0, 0, 0], dtype=np.float32)
    y = np.zeros((10000, 4), dtype=np.float32)
    for i in range(10000):
        y[i, :] = wiggle_quaternion(x, 0.1)
    avgy = y.mean(axis=0)
    avgy /= np.linalg.norm(avgy)
    assert np.allclose(avgy, x, rtol=0.001, atol=0.001)
