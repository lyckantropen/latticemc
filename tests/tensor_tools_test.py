import numpy as np

from latticemc.tensor_tools import (dot6, dot10, quaternion_to_orientation,
                                    t20_and_t22_in_6_coordinates,
                                    t20_t22_matrix, t32_in_10_coordinates,
                                    t32_matrix, ten6_to_mat)


def test_t20_and_t22_in6_coordinates():
    ex, ey, ez = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    t20_6, t22_6 = t20_and_t22_in_6_coordinates(ex, ey, ez)

    xx = np.outer(ex, ex)
    yy = np.outer(ey, ey)
    zz = np.outer(ez, ez)
    t20 = np.sqrt(3 / 2) * (zz - 1 / 3 * np.eye(3))
    t22 = np.sqrt(1 / 2) * (xx - yy)

    assert np.allclose(ten6_to_mat(t20_6), t20)
    assert np.allclose(ten6_to_mat(t22_6), t22)


def test_t32_in10_coordinates():
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
    t32_10 = t32_in_10_coordinates(ex, ey, ez)
    t32i_10 = t32_in_10_coordinates(tx, ty, tz)

    t32 = t32_matrix(ex, ey, ez)
    t32i = t32_matrix(tx, ty, tz)

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
    t20_6, t22_6 = t20_and_t22_in_6_coordinates(ex, ey, ez)
    t20i_6, t22i_6 = t20_and_t22_in_6_coordinates(tx, ty, tz)

    t20, t22 = t20_t22_matrix(ex, ey, ez)
    t20i, t22i = t20_t22_matrix(tx, ty, tz)

    q_6 = t20_6 + np.sqrt(2) * 0.3 * t22_6
    qi_6 = t20i_6 + np.sqrt(2) * 0.3 * t22i_6

    q = t20 + np.sqrt(2) * 0.3 * t22
    qi = t20i + np.sqrt(2) * 0.3 * t22i

    assert np.isclose(np.tensordot(q, qi), dot6(q_6, qi_6))


def test_quaternion_to_orientation():
    x = np.array([0.25091976, -0.90142864, -0.32465923, -0.13806562], dtype=np.float32)
    ex, ey, ez = quaternion_to_orientation(x)
    assert np.isclose(np.dot(ex, ey), 0, atol=1e-4)
    assert np.isclose(np.dot(ex, ez), 0, atol=1e-4)
    assert np.isclose(np.dot(ez, ey), 0, atol=1e-4)
    assert np.isclose(np.dot(ex, ex), 1, atol=1e-4)
    assert np.isclose(np.dot(ey, ey), 1, atol=1e-4)
    assert np.isclose(np.dot(ez, ez), 1, atol=1e-4)
