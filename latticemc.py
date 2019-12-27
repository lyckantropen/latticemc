import numpy as np
from definitions import particle, Lattice, LatticeState
from latticeTools import initializePartiallyOrdered
from statistical import fluctuation, ten6toMat
import simulationNumba


lattice = Lattice(9,9,9)
initializePartiallyOrdered(lattice)

state = LatticeState(temperature=1.186, lam=0.3, tau=1, lattice=lattice)
energyVariance = np.empty(1, np.float32)

for it in range(10000):
    simulationNumba.doLatticeStateUpdate(state)

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
