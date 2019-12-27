import numpy as np
from numba import njit
from statistical import fluctuation
from tensorTools import ten6toMat, dot10
from definitions import gatheredOrderParameters, LatticeState


def calculateOrderParameters(state: LatticeState):
    """
    Calculate instantaneous order parameters after
    the LatticeState has been updated.
    """
    avg = state.latticeAverages[-1]
    mQ = avg['t20'] + state.lam*np.sqrt(2)*avg['t22']
    mQ = ten6toMat(mQ)
    w = np.linalg.eigvalsh(mQ)
    # TODO: WHY THE MINUS???? Shouldn't be needed. Does LAPACK do something weird?
    wn, wm, wl = sorted(-w, key=lambda x: x**2, reverse=True)

    q0 = wn/(np.sqrt(2/3))
    q2 = (wl-wm)/(2/np.sqrt(2))

    energy = avg['energy']
    w = (q0**3 - 3*q0*q2**2)/(q0**2 + q2**2)**(3/2)
    p = avg['p']
    d322 = np.sqrt(dot10(avg['t32'], avg['t32']))

    return np.array([
        (energy, q0, q2, w, p, d322)
    ], dtype=gatheredOrderParameters)
