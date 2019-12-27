import numpy as np
from numba import njit
from itertools import product
from definitions import particle, Lattice, LatticeState
from randomQuaternion import randomQuaternion, wiggleQuaternion
import simulation_numba

def ten6toMat(a):
    ret = np.zeros((3,3), np.float32)
    ret[[0,1,2],[0,1,2]] = a[[0, 3, 5]]
    ret[[0,1],[1,0]] = a[1]
    ret[[0,2],[2,0]] = a[2]
    ret[[1,2],[2,1]] = a[4]
    return ret
    
@njit(cache=True)
def initializeRandomQuaternions(xlattice):
    for i in np.ndindex(xlattice.shape[:-1]):
        xlattice[i] = randomQuaternion(1)

@njit(cache=True)
def fluctuation(values):
    fluct = np.zeros_like(values)
    for i in range(fluct.size):
        e = np.random.choice(values, values.size)
        e2 = e*e
        fluct[i] = (e2.mean()-e.mean()**2)
    return fluct.mean()

lattice = Lattice(9,9,9)

##random state
# initializeRandomQuaternions(lattice.particles['x'])
# lattice.particles['p'] = np.random.choice([-1,1], L*L*L).reshape(L,L,L)

##ordered state
# lattice.particles['x'] = [1,0,0,0]
# lattice.particles['p'] = 1

##randomised partially ordered state
#<parity>~0.5, "mostly" biaxial nematic
lattice.particles['p'] = np.random.choice([-1,1], lattice.particles.size, p=[0.25,0.75]).reshape(lattice.particles.shape)
lattice.particles['x'] = [1,0,0,0]
for i in np.ndindex(lattice.particles.shape):
    lattice.particles['x'][i] = wiggleQuaternion(lattice.particles['x'][i], 0.02)

state = LatticeState(temperature=1.186, lam=0.3, tau=1, lattice=lattice)
energyVariance = np.empty(1, np.float32)

for it in range(10000):
    simulation_numba.doLatticeStateUpdate(state)

    if it > 100:
        energyVariance = np.append(energyVariance,lattice.particles.size*fluctuation(state.latticeAverages['energy'][-100:]))
    state.wiggleRateValues = np.append(state.wiggleRateValues, state.wiggleRate)

    # adjusting of wiggle rate
    if it % 10 == 0 and state.latticeAverages.size > 101:
        mE = np.array([ m.mean() for m in np.split(state.latticeAverages['energy'][-100:], 4)])
        mR = np.array([ m.mean() for m in np.split(state.wiggleRateValues[-100:], 4)])
        
        de = np.diff(mE, 1)
        dr = np.diff(mR, 1)
        efirst = de/dr
        etrend = (efirst[-1]-efirst[-2])
        if etrend < 0:
            state.wiggleRate *= 1.1
        else:
            state.wiggleRate *= 0.9
    state.wiggleRate *= 1 + np.random.normal(scale=0.001)
    
    # randomly adjust wiggle rate
    if it % 1000 == 0 and it > 0:
        state.wiggleRate = np.random.normal(state.wiggleRate, scale=1.0)

    # print stats
    if it % 50 == 0:
        mQ = state.latticeAverages['t20'][-1] + state.lam*np.sqrt(2)*state.latticeAverages['t22'][-1]
        mQ = ten6toMat(mQ)
        qw,qv = np.linalg.eig(mQ)

        q0 = np.sum(np.power(state.latticeAverages['t20'][-1],2))
        q2 = np.sum(np.power(state.latticeAverages['t22'][-1],2))
        p = state.latticeAverages['p'][-1]

        print(f'r={state.wiggleRate:.3f}, <E>={state.latticeAverages["energy"][-1]:.2f}, var(E)={energyVariance[-1]:.4f}, q0={q0:.6f}, q2={q2:.6f}, p={p:.4f}, qev={qw}')
