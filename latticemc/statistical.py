import numba as nb
import numpy as np
from nptyping import NDArray, Shape, Float32


@nb.njit(nb.float32(nb.float32[:]), cache=True)
def fluctuation(values: NDArray[Shape, Float32]) -> np.float32:
    """
    Calculate the fluctuation of one-dimensional array
    """
    fluct = np.zeros_like(values)
    for i in range(fluct.size):
        e = np.random.choice(values, values.size)
        e2 = e * e
        fluct[i] = (e2.mean() - e.mean()**2)
    return fluct.mean()
