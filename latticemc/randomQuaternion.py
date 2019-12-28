import numpy as np
from numba import njit


@njit(cache=True)
def randomQuaternion(wiggleRate):
    """
    Generate a random point on a 4D-sphere
    of radius equal to 'wiggleRate'
    """
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

    result[0] = wiggleRate * y1
    result[1] = wiggleRate * y2
    result[2] = wiggleRate * y3 * sr
    result[3] = wiggleRate * y4 * sr
    return result


@njit(cache=True)
def wiggleQuaternion(x, wiggleRate):
    """
    Return a normalised 4-vector that is offset from the previous
    one by a random 4-vector of radius 'wiggleRate'
    """
    dx = randomQuaternion(wiggleRate)
    x_ = x.copy()
    x_ += dx
    x_ /= np.linalg.norm(x_)
    return x_
