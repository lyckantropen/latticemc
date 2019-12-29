import numpy as np
from numba import jit, njit
from .tensorTools import ten6toMat, dot10, SQRT16
from .definitions import gatheredOrderParameters, LatticeState


@njit(cache=True)
def biaxialOrdering(lam):
    if lam < (SQRT16 - 1e-3):
        return 1
    if lam > (SQRT16 + 1e-3):
        return -1
    if (lam - SQRT16) < 1e-3:
        return 0


@njit(cache=True)
def _q0q2w(mt20, mt22, lam):
    mQ = mt20 + lam * np.sqrt(2) * mt22
    mQ = ten6toMat(mQ)
    ev = np.linalg.eigvalsh(mQ)
    i = np.argsort(ev**2)[::-1]
    wn, wm, wl = ev[i]

    nq0 = 1
    nq2 = 1
    o = biaxialOrdering(lam)
    if o == 1 or o == 0:
        nq0 = 1 / (np.sqrt(6) / 3)
        nq2 = 1 / (np.sqrt(2))
    if o == -1:
        nq0 = 1 / (np.sqrt(6) / 3 * 2)
        nq2 = 1 / (np.sqrt(2))

    q0 = wn * nq0
    q2 = (wl - wm) * nq2
    w = (q0**3 - 3 * q0 * q2**2) / (q0**2 + q2**2)**(3 / 2)
    return q0, q2, w


@njit(cache=True)
def _d322(mt32):
    return np.sqrt(dot10(mt32, mt32))


@jit(nopython=False, forceobj=True, parallel=True)
def calculateOrderParameters(state: LatticeState):
    """
    Calculate instantaneous order parameters after
    the LatticeState has been updated.
    """
    avg = state.latticeAverages[-1]
    q0, q2, w = _q0q2w(avg['t20'], avg['t22'], state.parameters.lam)

    energy = avg['energy']
    p = avg['p']
    d322 = _d322(avg['t32'])

    return np.array([
        (energy, q0, q2, w, p, d322)
    ], dtype=gatheredOrderParameters)
