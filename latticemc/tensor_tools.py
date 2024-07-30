from typing import Tuple

import numba as nb
import numpy as np
from nptyping import Float32, NDArray, Shape

SQRT2 = np.float32(np.sqrt(2))
SQRT6 = np.float32(np.sqrt(6))
SQRT32 = np.float32(np.sqrt(3 / 2))
SQRT12 = np.float32(np.sqrt(1 / 2))
SQRT16 = np.float32(np.sqrt(1 / 6))


@nb.njit(nb.types.UniTuple(nb.float32[:], 2)(nb.float32[:], nb.float32[:], nb.float32[:]), cache=True)
def t20_and_t22_in_6_coordinates(ex: NDArray[Shape['3'], Float32],
                                 ey: NDArray[Shape['3'], Float32],
                                 ez: NDArray[Shape['3'], Float32]
                                 ) -> Tuple[NDArray[Shape['6'], Float32], NDArray[Shape['6'], Float32]]:
    """Calculate the T20 and T22 tensors in the coordinate system defined by `ex`, `ey`, `ez` using only 6 independent components."""
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
    t20 = (SQRT32 * (zz - 1 / 3 * ident)).astype(np.float32)
    t22 = (SQRT12 * (xx - yy)).astype(np.float32)

    return t20, t22


@nb.njit(nb.float32[:](nb.float32[:], nb.float32[:], nb.float32[:]), cache=True)
def t32_in_10_coordinates(ex: NDArray[Shape['3'], Float32],
                          ey: NDArray[Shape['3'], Float32],
                          ez: NDArray[Shape['3'], Float32]
                          ) -> NDArray[Shape['10'], Float32]:
    """Calculate the T32 tensor in the coordinate system defined by `ex`, `ey`, `ez` using only 10 independent components."""
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

    return t32 * SQRT16


@nb.njit(nb.float32[:, :](nb.float32[:]), cache=True)
def ten6_to_mat(a: NDArray[Shape['6'], Float32]) -> NDArray[Shape['3,3'], Float32]:
    """Convert a symmetric tensor represented using 6 independent components to a 3x3 matrix."""
    ret = np.zeros((3, 3), np.float32)
    ret[0, 0] = a[0]
    ret[1, 1] = a[3]
    ret[2, 2] = a[5]
    ret[0, 1] = ret[1, 0] = a[1]
    ret[0, 2] = ret[2, 0] = a[2]
    ret[1, 2] = ret[2, 1] = a[4]
    return ret


@nb.njit(nb.float32(nb.float32[:], nb.float32[:]), cache=True)
def dot6(a: NDArray[Shape['6'], Float32], b: NDArray[Shape['6'], Float32]) -> np.float32:
    """Perform rank-2 contraction using only the 6 non-zero coefficients of a symmetric tensor."""
    return (a[0] * b[0] +
            a[3] * b[3] +
            a[5] * b[5] +
            2.0 * a[1] * b[1] +
            2.0 * a[2] * b[2] +
            2.0 * a[4] * b[4])


@nb.njit(nb.float32(nb.float32[:], nb.float32[:]), cache=True)
def dot10(a: NDArray[Shape['10'], Float32], b: NDArray[Shape['10'], Float32]) -> np.float32:
    """Perform rank-3 contraction using only the 10 non-zero coefficients of a symmetric tensor."""
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


@nb.njit(nb.types.UniTuple(nb.float32[:], 3)(nb.float32[:]), cache=True)
def quaternion_to_orientation(x: NDArray[Shape['4'], Float32]) -> Tuple[NDArray[Shape['3'], Float32], NDArray[Shape['3'], Float32], NDArray[Shape['3'], Float32]]:
    """
    Convert an arbitrary normalized quaternion to a proper rotation in 3D space.

    Returns
    -------
    Tuple[NDArray[(3,), np.float32], NDArray[(3,), np.float32], NDArray[(3,), np.float32]]
        three orthonormal vectors defining the target coordinate system
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


@nb.njit(nb.types.UniTuple(nb.float32[:, :], 2)(nb.float32[:], nb.float32[:], nb.float32[:]), cache=True)
def t20_t22_matrix(ex: NDArray[Shape['3'], Float32],
                   ey: NDArray[Shape['3'], Float32],
                   ez: NDArray[Shape['3'], Float32]
                   ) -> Tuple[NDArray[Shape['3,3'], Float32], NDArray[Shape['3,3'], Float32]]:
    """Create the T20 and T22 tensors as 3x3 matrices in the coordinate system defined by `ex`, `ey` and `ez."""
    xx = np.outer(ex, ex)
    yy = np.outer(ey, ey)
    zz = np.outer(ez, ez)
    t20 = SQRT32 * (zz - 1 / 3 * np.eye(3)).astype(np.float32)
    t22 = SQRT12 * (xx - yy).astype(np.float32)
    return t20, t22


@nb.njit(nb.float32[:, :, :](nb.float32[:], nb.float32[:], nb.float32[:]), cache=True)
def t32_matrix(ex: NDArray[Shape['3'], Float32],
               ey: NDArray[Shape['3'], Float32],
               ez: NDArray[Shape['3'], Float32]) -> NDArray[Shape['3, 3, 3'], Float32]:
    """Create the T32 tensor as a 3x3x3 matrix in the coordinate system defined by `ex`, `ey` and `ez."""
    return SQRT16 * (
        np.outer(np.outer(ex, ey), ez) +
        np.outer(np.outer(ez, ex), ey) +
        np.outer(np.outer(ey, ez), ex) +
        np.outer(np.outer(ex, ez), ey) +
        np.outer(np.outer(ey, ex), ez) +
        np.outer(np.outer(ez, ey), ex)
    ).reshape(3, 3, 3).astype(np.float32)
