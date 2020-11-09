import numpy as np
from latticemc.orderParameters import _d322, _q0q2w
from latticemc.tensorTools import T20AndT22In6Coordinates, T32In10Coordinates


def test__q0q2w():
    a, b, c = (
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    )

    t20x, t22x = T20AndT22In6Coordinates(a, b, c)
    t20y, t22y = T20AndT22In6Coordinates(c, a, b)
    t20z, t22z = T20AndT22In6Coordinates(b, c, a)

    q0x, q2x, wx = _q0q2w(t20x, t22x, 0)
    q0y, q2y, wy = _q0q2w(t20y, t22y, np.sqrt(3 / 2))
    q0z, q2z, wz = _q0q2w(t20z, t22z, np.sqrt(1 / 6))

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
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    )
    t32 = T32In10Coordinates(a, b, c)
    d322 = _d322(t32)
    assert np.isclose(d322, 1)
