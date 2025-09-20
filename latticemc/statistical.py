import numba as nb
import numpy as np
from jaxtyping import Float32


@nb.njit(nb.float32(nb.float32[:]), cache=True)
def fluctuation(values: Float32[np.ndarray, "n"]) -> np.float32:
    """
    Calculate the bootstrap variance (fluctuation) of one-dimensional array

    This function estimates the variance using bootstrap resampling:
    - For each bootstrap iteration, it randomly samples (with replacement)
      from the input values
    - Calculates the variance of each bootstrap sample
    - Returns the mean of all bootstrap variances
    """
    fluct = np.zeros_like(values)
    n = values.size

    for i in range(fluct.size):
        # Bootstrap resample: random indices for sampling with replacement
        indices = np.random.randint(0, n, n)
        e = values[indices]  # Use indexing instead of np.random.choice for numba
        e2 = e * e
        fluct[i] = (e2.mean() - e.mean()**2)

    return fluct.mean()
