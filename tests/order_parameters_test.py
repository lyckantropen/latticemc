import numpy as np

from latticemc.order_parameters import _d322, _q0q2w
from latticemc.tensor_tools import (t20_and_t22_in_6_coordinates,
                                    t32_in_10_coordinates)


def test__q0q2w():
    a, b, c = (
        np.array([1, 0, 0], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32),
        np.array([0, 0, 1], dtype=np.float32)
    )

    t20x, t22x = t20_and_t22_in_6_coordinates(a, b, c)
    t20y, t22y = t20_and_t22_in_6_coordinates(c, a, b)
    t20z, t22z = t20_and_t22_in_6_coordinates(b, c, a)

    q0x, q2x, wx = _q0q2w(t20x, t22x, 0)
    q0y, q2y, wy = _q0q2w(t20y, t22y, np.sqrt(3 / 2))
    q0z, q2z, wz = _q0q2w(t20z, t22z, np.sqrt(1 / 6) + 1e-7)

    assert np.isclose(q0x, 1, atol=1e-6)
    assert np.isclose(q2x, 0, atol=1e-6)
    assert np.isclose(wx, 1, atol=1e-6)

    assert np.isclose(q0y, -1, atol=1e-6)
    assert np.isclose(q2y, 0, atol=1e-6)
    assert np.isclose(wy, -1, atol=1e-6)

    assert np.isclose(q0z, -1, atol=1e-6)
    assert np.isclose(q2z, -np.sqrt(3) / 3, atol=1e-6)
    assert np.isclose(wz, 0, atol=1e-6)


def test__d322():
    a, b, c = (
        np.array([1, 0, 0], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32),
        np.array([0, 0, 1], dtype=np.float32)
    )
    t32 = t32_in_10_coordinates(a, b, c)
    d322 = _d322(t32)
    assert np.isclose(d322, 1)
