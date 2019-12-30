from latticemc.definitions import Lattice, LatticeState, OrderParametersHistory, DefiningParameters
from latticemc.latticeTools import initializePartiallyOrdered
from latticemc.randomQuaternion import randomQuaternion
from latticemc.updaters import OrderParametersCalculator, FluctuationsCalculator, DerivativeWiggleRateAdjustor, RandomWiggleRateAdjustor
from latticemc.failsafe import failsafeSaveSimulation
from latticemc import simulationNumba
from joblib import Parallel, delayed
import numpy as np

temperatures = np.arange(0.2, 1, 0.1)
states = [LatticeState(DefiningParameters(temperature=t, lam=0.3, tau=1), lattice=Lattice(9, 9, 9)) for t in temperatures]
orderParametersHistory = {s.parameters: OrderParametersHistory() for s in states}
for s in states:
    initializePartiallyOrdered(s.lattice, x=randomQuaternion(1))


def doState(state):
    try:
        localHistory = {state.parameters: OrderParametersHistory()}
        orderParametersCalculator = OrderParametersCalculator(localHistory, howOften=1, sinceWhen=1, printEvery=50)
        fluctuationsCalculator = FluctuationsCalculator(localHistory, window=100, howOften=50, sinceWhen=100, printEvery=50)
        perStateUpdaters = [
            orderParametersCalculator,
            fluctuationsCalculator,
            DerivativeWiggleRateAdjustor(howMany=100, howOften=10, sinceWhen=101),
            RandomWiggleRateAdjustor(scale=0.001, howOften=10, sinceWhen=1),
            RandomWiggleRateAdjustor(scale=1.0, resetValue=1.0, howOften=1000, sinceWhen=1000),
        ]
        for it in range(10):
            simulationNumba.doLatticeStateUpdate(state)
            for u in perStateUpdaters:
                u.perform(state)
    except Exception as e:
        failsafeSaveSimulation(e, state, orderParametersHistory)
    return state, localHistory


with Parallel(n_jobs=len(states), backend='loky') as parallel:
    for bigIt in range(100000):
        results = parallel(delayed(doState)(state) for state in states)
        states, _ = list(zip(*results))
        for state, hist in results:
            p = state.parameters
            orderParametersHistory[p].orderParameters = np.append(orderParametersHistory[p].orderParameters, hist[p].orderParameters)
            orderParametersHistory[p].fluctuations = np.append(orderParametersHistory[p].fluctuations, hist[p].fluctuations)
        for _ in range(0, len(states)):
            i = np.random.randint(0, len(states) - 1)
            s1, s2 = states[i], states[i + 1]
            dE = (orderParametersHistory[s1.parameters].orderParameters['energy'][-1] -
                  orderParametersHistory[s2.parameters].orderParameters['energy'][-1])
            dB = 1 / s1.parameters.temperature - 1 / s2.parameters.temperature
            print(f'dE={dE}, dB={dB}, e={np.exp(dE*dB)}')
            if np.random.random() < np.exp(dE * dB):
                t1, t2 = s1.parameters.temperature, s2.parameters.temperature
                s1.parameters = DefiningParameters(temperature=t2, lam=s1.parameters.lam, tau=s1.parameters.tau)
                s2.parameters = DefiningParameters(temperature=t1, lam=s2.parameters.lam, tau=s2.parameters.tau)
                print(f'Exchanged {i} and {i+1}')
