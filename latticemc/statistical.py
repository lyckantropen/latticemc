import numpy as np
from numba import njit


@njit(cache=True)
def fluctuation(values):
    """
    Calculate the fluctuation of one-dimensional array
    """
    fluct = np.zeros_like(values)
    for i in range(fluct.size):
        e = np.random.choice(values, values.size)
        e2 = e*e
        fluct[i] = (e2.mean()-e.mean()**2)
    return fluct.mean()
