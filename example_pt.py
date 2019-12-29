from latticemc.definitions import Lattice, LatticeState, OrderParametersHistory, DefiningParameters
from latticemc.latticeTools import initializePartiallyOrdered
from latticemc.randomQuaternion import randomQuaternion
from latticemc.updaters import OrderParametersCalculator, FluctuationsCalculator, DerivativeWiggleRateAdjustor, RandomWiggleRateAdjustor
from latticemc.failsafe import failsafeSaveSimulation
from latticemc import simulationNumba
from joblib import Parallel, delayed
import numpy as np

temperatures = np.arange(0.2, 2.2, 0.1)
states = [LatticeState(DefiningParameters(temperature=t, lam=0.3, tau=1), lattice=Lattice(9, 9, 9)) for t in temperatures]
orderParametersHistory = {s.parameters: OrderParametersHistory() for s in states}
for s in states:
    initializePartiallyOrdered(s.lattice, x=randomQuaternion(1))


orderParametersCalculator = OrderParametersCalculator(orderParametersHistory, howOften=1, sinceWhen=1, printEvery=50)
fluctuationsCalculator = FluctuationsCalculator(orderParametersHistory, window=100, howOften=50, sinceWhen=100, printEvery=50)
perStateUpdaters = [
    orderParametersCalculator,
    fluctuationsCalculator,
    DerivativeWiggleRateAdjustor(howMany=100, howOften=10, sinceWhen=101),
    RandomWiggleRateAdjustor(scale=0.001, howOften=10, sinceWhen=1),
    RandomWiggleRateAdjustor(scale=1.0, resetValue=1.0, howOften=1000, sinceWhen=1000),
]


def doState(state):
    try:
        for it in range(10000):
            simulationNumba.doLatticeStateUpdate(state)
            for u in perStateUpdaters:
                u.perform(state)
    except Exception as e:
        failsafeSaveSimulation(e, state, orderParametersHistory)
    return True


with Parallel(n_jobs=len(states)) as parallel:
    results = parallel(delayed(doState)(state) for state in states)
