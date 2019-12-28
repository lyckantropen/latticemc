from . import disableNumba  # noqa: F401
import numpy as np
from latticemc.tensorTools import (
    T20AndT22In6Coordinates,
    T32In10Coordinates,
    ten6toMat,
    dot6,
    dot10,
    quaternionToOrientation
)


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


def test_T20AndT22In6Coordinates():
    ex, ey, ez = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    t20_6, t22_6 = T20AndT22In6Coordinates(ex, ey, ez)

    xx = np.outer(ex, ex)
    yy = np.outer(ey, ey)
    zz = np.outer(ez, ez)
    t20 = np.sqrt(3 / 2) * (zz - 1 / 3 * np.eye(3))
    t22 = np.sqrt(1 / 2) * (xx - yy)

    assert np.allclose(ten6toMat(t20_6), t20)
    assert np.allclose(ten6toMat(t22_6), t22)


def test_T32In10Coordinates():
    ex, ey, ez = (
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    )
    tx, ty, tz = (
        np.array([0.75106853, 0.5160275, 0.41183946]),
        np.array([0.65460104, -0.6632714, -0.36272395]),
        np.array([0.08598578, 0.5420211, -0.8359544])
    )
    t32_10 = T32In10Coordinates(ex, ey, ez)
    t32i_10 = T32In10Coordinates(tx, ty, tz)

    t32 = _t32Matrix(ex, ey, ez)
    t32i = _t32Matrix(tx, ty, tz)

    assert np.isclose(np.tensordot(t32, t32i, axes=3), dot10(t32_10, t32i_10), atol=1e-4)


def test_dot6():
    ex, ey, ez = (
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    )
    tx, ty, tz = (
        np.array([0.75106853, 0.5160275, 0.41183946]),
        np.array([0.65460104, -0.6632714, -0.36272395]),
        np.array([0.08598578, 0.5420211, -0.8359544])
    )
    t20_6, t22_6 = T20AndT22In6Coordinates(ex, ey, ez)
    t20i_6, t22i_6 = T20AndT22In6Coordinates(tx, ty, tz)

    t20, t22 = _t20t22Matrix(ex, ey, ez)
    t20i, t22i = _t20t22Matrix(tx, ty, tz)

    Q_6 = t20_6 + np.sqrt(2) * 0.3 * t22_6
    Qi_6 = t20i_6 + np.sqrt(2) * 0.3 * t22i_6

    Q = t20 + np.sqrt(2) * 0.3 * t22
    Qi = t20i + np.sqrt(2) * 0.3 * t22i

    assert np.isclose(np.tensordot(Q, Qi), dot6(Q_6, Qi_6))


def test_quaternionToOrientation():
    x = np.array([0.25091976, -0.90142864, -0.32465923, -0.13806562], dtype=np.float32)
    ex, ey, ez = quaternionToOrientation(x)
    assert np.isclose(np.dot(ex, ey), 0, atol=1e-4)
    assert np.isclose(np.dot(ex, ez), 0, atol=1e-4)
    assert np.isclose(np.dot(ez, ey), 0, atol=1e-4)
    assert np.isclose(np.dot(ex, ex), 1, atol=1e-4)
    assert np.isclose(np.dot(ey, ey), 1, atol=1e-4)
    assert np.isclose(np.dot(ez, ez), 1, atol=1e-4)
