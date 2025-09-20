from typing import Tuple

import numba as nb
import numpy as np
from jaxtyping import Float32, Shaped

from .definitions import LatticeState, gathered_order_parameters
from .tensor_tools import SQRT2, SQRT6, SQRT16, dot10, ten6_to_mat


@nb.njit(nb.int32(nb.float32), cache=True)
def biaxial_ordering(lam: float) -> int:
    if lam < (SQRT16 - 1e-3):
        return 1
    if lam > (SQRT16 + 1e-3):
        return -1
    if abs(lam - SQRT16) < 1e-3:
        return 0
    return -100


@nb.njit(nb.types.UniTuple(nb.float32, 3)(nb.float32[:], nb.float32[:], nb.float32), cache=True)
def _q0q2w(mt20: Float32[np.ndarray, "6"],
           mt22: Float32[np.ndarray, "6"],
           lam: np.float32
           ) -> Tuple[np.float32, np.float32, np.float32]:
    m_q = mt20 + lam * SQRT2 * mt22
    m_q = ten6_to_mat(m_q)
    ev = np.linalg.eigvalsh(m_q)
    i = np.argsort(ev**2)[::-1]
    wn, wm, wl = ev[i]

    nq0 = np.float32(1)
    nq2 = np.float32(1)
    o = biaxial_ordering(lam)
    if o == 1 or o == 0:
        nq0 = np.float32(1 / (SQRT6 / 3))
        nq2 = np.float32(1 / SQRT2)
    if o == -1:
        nq0 = np.float32(1 / (SQRT6 / 3 * 2))
        nq2 = np.float32(1 / SQRT2)

    q0 = wn * nq0
    q2 = (wl - wm) * nq2
    w = (q0**3 - 3 * q0 * q2**2) / (q0**2 + q2**2)**(3 / 2)
    return q0, q2, w


@nb.njit(nb.float32(nb.float32[:]), cache=True)
def _d322(mt32: Float32[np.ndarray, "10"]) -> np.float32:
    return np.sqrt(dot10(mt32, mt32))


@nb.jit(nopython=False, forceobj=True, parallel=True)
def calculate_order_parameters(state: LatticeState) -> Shaped[np.ndarray, "..."]:
    """Calculate instantaneous order parameters after the `LatticeState` has been updated."""
    avg = state.lattice_averages[0]
    q0, q2, w = _q0q2w(avg['t20'], avg['t22'], float(state.parameters.lam))

    energy = avg['energy']
    p = avg['p']
    d322 = _d322(avg['t32'])

    return np.array([
        (energy, q0, q2, w, p, d322)
    ], dtype=gathered_order_parameters)
