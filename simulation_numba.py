"""
Monte Carlo lattice simulation accelerated using Numba
(much, much faster than plain Python)
"""
import numpy as np
from numba import jit,njit
from randomQuaternion import randomQuaternion, wiggleQuaternion
from definitions import particle, Lattice, LatticeState


@njit(cache=True)
def _dot6(a, b):
    """
    Rank-2 contraction using only the
    6 non-zero coefficients of a symmetric
    tensor.
    """
    return (a[0]*b[0]+
            a[3]*b[3]+
            a[5]*b[5]+
            2.0*a[1]*b[1]+
            2.0*a[2]*b[2]+
            2.0*a[4]*b[4])

@njit(cache=True)
def _dot10(a, b):
    """
    Rank-3 contraction using only the
    10 non-zero coefficients of a symmetric
    tensor.
    """
    coeff = np.zeros(10, np.float32)
    coeff[0]=1
    coeff[1]=3
    coeff[2]=3
    coeff[3]=1
    coeff[4]=3
    coeff[5]=6
    coeff[6]=3
    coeff[7]=3
    coeff[8]=3
    coeff[9]=1
    coeff*=a*b
    return coeff.sum()

@njit(cache=True)
def _getPropertiesFromOrientation(x, parity):
    x11 = x[1] * x[1]
    x22 = x[2] * x[2]
    x33 = x[3] * x[3]
    x01 = x[0] * x[1]
    x02 = x[0] * x[2]
    x03 = x[0] * x[3]
    x12 = x[1] * x[2]
    x13 = x[1] * x[3]
    x23 = x[2] * x[3]

    ex = [2 * (-x22 - x33 + 0.5), 2 * (x12 + x03), 2 * (x13 - x02)]
    ey = [2 * (x12 - x03), 2 * (-x11 - x33 + 0.5), 2 * (x01 + x23)]
    ez = [2 * (x02 + x13), 2 * (-x01 + x23), 2 * (-x22 - x11 + 0.5)]

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

    I = np.zeros(6, np.float32)
    I[np.array([0,3,5])] = 1
    t20 = np.sqrt(3/2)*(zz - 1/3*I)
    t22 = np.sqrt(1/2)*(xx - yy)

    t32 = np.zeros(10, np.float32)

    #000
    t32[0] = 6.0 * (ex[0] * ey[0] * ez[0]) # 1
    #100
    t32[1] = 2.0 * (ex[0] * ey[0] * ez[1] + # 3
                    ex[0] * ey[1] * ez[0] +
                    ex[1] * ey[0] * ez[0])
    #110
    t32[2] = 2.0 * (ex[0] * ey[1] * ez[1] + # 3
                    ex[1] * ey[0] * ez[1] +
                    ex[1] * ey[1] * ez[0])
    #111
    t32[3] = 6.0 * (ex[1] * ey[1] * ez[1]) # 1
    #200
    t32[4] = 2.0 * (ex[0] * ey[0] * ez[2] + # 3
                    ex[0] * ey[2] * ez[0] +
                    ex[2] * ey[0] * ez[0])
    #210
    t32[5] = (ex[0] * ey[1] * ez[2] +     # 6
              ex[0] * ey[2] * ez[1] +
              ex[1] * ey[0] * ez[2] +
              ex[2] * ey[0] * ez[1] +
              ex[1] * ey[2] * ez[0] +
              ex[2] * ey[1] * ez[0]
             )
    #211
    t32[6] = 2.0 * (ex[1] * ey[1] * ez[2] + # 3
                    ex[1] * ey[2] * ez[1] +
                    ex[2] * ey[1] * ez[1]
                   )
    #220
    t32[7] = 2.0 * (ex[2] * ey[2] * ez[0] + # 3
                    ex[0] * ey[2] * ez[2] +
                    ex[2] * ey[0] * ez[2])
    #221
    t32[8] = 2.0 * (ex[2] * ey[2] * ez[1] + # 3
                    ex[1] * ey[2] * ez[2] +
                    ex[2] * ey[1] * ez[2])
    #222
    t32[9] = 6.0 * (ex[2] * ey[2] * ez[2]) # 1

    t32*=(parity/np.sqrt(6))

    return t20, t22, t32

@njit(cache=True)
def _getNeighbors(center, lattice):
    ind = np.mod(np.array([
        [ 1,0,0],
        [-1,0,0],
        [0, 1,0],
        [0,-1,0],
        [0,0, 1],
        [0,0,-1]
    ]) + center, lattice.shape[0])
    n = np.zeros((6, lattice.shape[-1]), lattice.dtype)
    n[0] = lattice[ind[0,0], ind[0,1], ind[0,2]]
    n[1] = lattice[ind[1,0], ind[1,1], ind[1,2]]
    n[2] = lattice[ind[2,0], ind[2,1], ind[2,2]]
    n[3] = lattice[ind[3,0], ind[3,1], ind[3,2]]
    n[4] = lattice[ind[4,0], ind[4,1], ind[4,2]]
    n[5] = lattice[ind[5,0], ind[5,1], ind[5,2]]
    return n


@njit(cache=True)
def _getEnergy(x, p, nx, npi, lam, tau):
    t20, t22, t32 = _getPropertiesFromOrientation(x, p)
    Q = t20 + lam*np.sqrt(2)*t22
    energy = 0
    for i in range(nx.shape[0]):
        t20i, t22i, t32i = _getPropertiesFromOrientation(nx[i], npi[i])
        Qi = t20i + lam*np.sqrt(2)*t22i
        energy += (-_dot6(Q,Qi)-tau*_dot10(t32,t32i))/2
    return energy

@njit(cache=True)
def _metropolis(dE, temperature):
    if dE < 0:
        return True
    else:
        if np.random.random() < np.exp(-dE/temperature):
            return True
    return False

@jit(forceobj=True,nopython=False,parallel=True)
def _doOrientationSweep(lattice, indexes, temperature, lam, tau, wiggleRate):
    for _i in indexes:
        particle = lattice[tuple(_i)]
        nx = _getNeighbors(_i, lattice['x'])
        npi = _getNeighbors(_i, lattice['p'][...,np.newaxis])
        energy1 = _getEnergy(particle['x'], particle['p'], nx, npi, lam=lam, tau=tau)
        
        # adjust x
        x_ = wiggleQuaternion(particle['x'], wiggleRate)
        energy2 = _getEnergy(x_, particle['p'], nx, npi, lam=lam, tau=tau)
        if _metropolis(2*(energy2-energy1), temperature):
            particle['x'] = x_
            particle['energy'] = energy2

        # adjust p
        p_ = -particle['p']
        energy2 = _getEnergy(particle['x'], p_, nx, npi, lam=lam, tau=tau)
        if _metropolis(2*(energy2-energy1), temperature):
            particle['p'] = p_
            particle['energy'] = energy2

        particle['t20'], particle['t22'], particle['t32'] = _getPropertiesFromOrientation(particle['x'], particle['p'])

@jit(forceobj=True,nopython=False,cache=True)
def _getLatticeAverages(lattice):
    avg = np.zeros(1, dtype=particle)
    avg['x'] = lattice['x'].mean(axis=(0,1,2))
    avg['t20'] = lattice['t20'].mean(axis=(0,1,2))
    avg['t22'] = lattice['t22'].mean(axis=(0,1,2))
    avg['t32'] = lattice['t32'].mean(axis=(0,1,2))
    avg['p'] = lattice['p'].mean()
    avg['energy'] = lattice['energy'].mean()
    return avg[0]

@jit(forceobj=True,nopython=False,cache=True)
def doLatticeStateUpdate(state: LatticeState):
    # update particles at random
    indexes = state.lattice.particles['index'].reshape(-1,3)[np.random.randint(0, state.lattice.particles.size, state.lattice.particles.size)]
    _doOrientationSweep(state.lattice.particles, indexes, state.temperature, state.lam, state.tau, state.wiggleRate)
    state.iterations += 1
    state.latticeAverages = np.append(state.latticeAverages, _getLatticeAverages(state.lattice.particles))
