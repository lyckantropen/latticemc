import numpy as np
from definitions import particle, Lattice, LatticeState, gatheredOrderParameters
from latticeTools import initializePartiallyOrdered, initializeRandom
from statistical import fluctuation
from orderParameters import calculateOrderParameters
from randomQuaternion import randomQuaternion
import simulationNumba


lattice = Lattice(9,9,9)
initializePartiallyOrdered(lattice, x=randomQuaternion(1))
#initializeRandom(lattice)

state = LatticeState(temperature=0.9, lam=0.3, tau=1, lattice=lattice)
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
        op = calculateOrderParameters(state)
        s = ','.join([ f'{name}={op[name][0]}' for name in gatheredOrderParameters.fields.keys()])

        print(s)