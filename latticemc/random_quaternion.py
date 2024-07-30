import numba as nb
import numpy as np
from nptyping import NDArray, Shape, Float32


@nb.njit(nb.float32[:](nb.float32), cache=True)
def random_quaternion(radius: np.float32) -> NDArray[Shape['4'], Float32]:
    """Generate a random point on a 4D-sphere of radius equal to 'wiggle_rate'."""
    result = np.zeros(4, dtype=np.float32)

    r1 = 0
    r2 = 0
    (y1, y2, y3, y4) = (0, 0, 0, 0)
    while True:
        y1 = 1.0 - 2.0 * np.random.random()
        y2 = 1.0 - 2.0 * np.random.random()
        r1 = y1**2 + y2**2
        if r1 <= 1:
            break

    while True:
        y3 = 1.0 - 2.0 * np.random.random()
        y4 = 1.0 - 2.0 * np.random.random()
        r2 = y3**2 + y4**2
        if r2 <= 1:
            break

    sr = np.sqrt((1 - r1) / r2)

    result[0] = radius * y1
    result[1] = radius * y2
    result[2] = radius * y3 * sr
    result[3] = radius * y4 * sr
    return result


@nb.njit(nb.float32[:](nb.float32[:], nb.float32), cache=True)
def wiggle_quaternion(x: NDArray[Shape['4'], Float32], wiggle_rate: np.float32) -> NDArray[Shape['4'], Float32]:
    """Return a normalised 4-vector that is offset from the previous one by a random 4-vector of radius 'wiggle_rate'."""
    dx = random_quaternion(wiggle_rate)
    x_ = x.copy()
    x_ += dx
    x_ /= np.linalg.norm(x_)
    return x_
