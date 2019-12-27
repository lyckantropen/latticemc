import numpy as np
from numba import njit

def ten6toMat(a):
    ret = np.zeros((3,3), np.float32)
    ret[[0,1,2],[0,1,2]] = a[[0, 3, 5]]
    ret[[0,1],[1,0]] = a[1]
    ret[[0,2],[2,0]] = a[2]
    ret[[1,2],[2,1]] = a[4]
    return ret

@njit(cache=True)
def fluctuation(values):
    fluct = np.zeros_like(values)
    for i in range(fluct.size):
        e = np.random.choice(values, values.size)
        e2 = e*e
        fluct[i] = (e2.mean()-e.mean()**2)
    return fluct.mean()