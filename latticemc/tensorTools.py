import numpy as np
from numba import njit

SQRT2 = np.sqrt(2)
SQRT6 = np.sqrt(6)
SQRT32 = np.sqrt(3 / 2)
SQRT12 = np.sqrt(1 / 2)
SQRT16 = np.sqrt(1 / 6)


@njit(cache=True)
def T20AndT22In6Coordinates(ex, ey, ez):
    """
    Calculate the T20 and T22 tensors using only 6
    independent components.
    """
    xx = np.zeros(6, np.float32)
    yy = np.zeros(6, np.float32)
    zz = np.zeros(6, np.float32)

    xx[0] = ex[0] * ex[0]
    xx[1] = ex[0] * ex[1]
    xx[2] = ex[0] * ex[2]
    xx[3] = ex[1] * ex[1]
    xx[4] = ex[1] * ex[2]
    xx[5] = ex[2] * ex[2]

    yy[0] = ey[0] * ey[0]
    yy[1] = ey[0] * ey[1]
    yy[2] = ey[0] * ey[2]
    yy[3] = ey[1] * ey[1]
    yy[4] = ey[1] * ey[2]
    yy[5] = ey[2] * ey[2]

    zz[0] = ez[0] * ez[0]
    zz[1] = ez[0] * ez[1]
    zz[2] = ez[0] * ez[2]
    zz[3] = ez[1] * ez[1]
    zz[4] = ez[1] * ez[2]
    zz[5] = ez[2] * ez[2]

    ident = np.zeros(6, np.float32)
    ident[0] = 1
    ident[3] = 1
    ident[5] = 1
    t20 = SQRT32 * (zz - 1 / 3 * ident)
    t22 = SQRT12 * (xx - yy)

    return t20, t22


def ten6toMat(a):
    """
    Convert a symmetric tensor represented using 6
    independent components to a 3x3 matrix
    """
    ret = np.zeros((3, 3), np.float32)
    ret[[0, 1, 2], [0, 1, 2]] = a[[0, 3, 5]]
    ret[[0, 1], [1, 0]] = a[1]
    ret[[0, 2], [2, 0]] = a[2]
    ret[[1, 2], [2, 1]] = a[4]
    return ret


@njit(cache=True)
def dot6(a, b):
    """
    Rank-2 contraction using only the
    6 non-zero coefficients of a symmetric
    tensor.
    """
    return (a[0] * b[0] +
            a[3] * b[3] +
            a[5] * b[5] +
            2.0 * a[1] * b[1] +
            2.0 * a[2] * b[2] +
            2.0 * a[4] * b[4])


@njit(cache=True)
def dot10(a, b):
    """
    Rank-3 contraction using only the
    10 non-zero coefficients of a symmetric
    tensor.
    """
    coeff = np.zeros(10, np.float32)
    coeff[0] = 1
    coeff[1] = 3
    coeff[2] = 3
    coeff[3] = 1
    coeff[4] = 3
    coeff[5] = 6
    coeff[6] = 3
    coeff[7] = 3
    coeff[8] = 3
    coeff[9] = 1
    coeff *= a * b
    return coeff.sum()


@njit(cache=True)
def quaternionToOrientation(x):
    """
    Convert arbitrary normalized quaternion to
    a proper rotation in 3D space.
    """
    x11 = x[1] * x[1]
    x22 = x[2] * x[2]
    x33 = x[3] * x[3]
    x01 = x[0] * x[1]
    x02 = x[0] * x[2]
    x03 = x[0] * x[3]
    x12 = x[1] * x[2]
    x13 = x[1] * x[3]
    x23 = x[2] * x[3]

    ex = np.array([2 * (-x22 - x33 + 0.5), 2 * (x12 + x03), 2 * (x13 - x02)], dtype=np.float32)
    ey = np.array([2 * (x12 - x03), 2 * (-x11 - x33 + 0.5), 2 * (x01 + x23)], dtype=np.float32)
    ez = np.array([2 * (x02 + x13), 2 * (-x01 + x23), 2 * (-x22 - x11 + 0.5)], dtype=np.float32)
    return ex, ey, ez
