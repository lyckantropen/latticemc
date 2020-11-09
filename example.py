from decimal import Decimal

from latticemc import simulationNumba
from latticemc.definitions import (DefiningParameters, Lattice, LatticeState,
                                   OrderParametersHistory)
from latticemc.failsafe import failsafeSaveSimulation
from latticemc.latticeTools import initializePartiallyOrdered
from latticemc.randomQuaternion import randomQuaternion
from latticemc.updaters import (DerivativeWiggleRateAdjustor,
                                FluctuationsCalculator,
                                OrderParametersCalculator,
                                RandomWiggleRateAdjustor)

lattice = Lattice(9, 9, 9)
initializePartiallyOrdered(lattice, x=randomQuaternion(1))
# initializeRandom(lattice)

modelParams = DefiningParameters(temperature=round(Decimal(0.9), 1), lam=round(Decimal(0.3), 1), tau=round(Decimal(1), 1))
state = LatticeState(parameters=modelParams, lattice=lattice)
orderParametersHistory = OrderParametersHistory()

orderParametersCalculator = OrderParametersCalculator(orderParametersHistory, howOften=1, sinceWhen=1, printEvery=50)
fluctuationsCalculator = FluctuationsCalculator(orderParametersHistory, window=100, howOften=50, sinceWhen=100, printEvery=50)
updaters = [
    orderParametersCalculator,
    fluctuationsCalculator,
    DerivativeWiggleRateAdjustor(orderParametersHistory, howMany=100, howOften=10, sinceWhen=101),
    RandomWiggleRateAdjustor(scale=0.001, howOften=10, sinceWhen=1),
    RandomWiggleRateAdjustor(scale=1.0, resetValue=1.0, howOften=1000, sinceWhen=1000),
]

try:
    for it in range(10000):
        simulationNumba.doLatticeStateUpdate(state)

        for u in updaters:
            u.perform(state)

except Exception as e:
    failsafeSaveSimulation(e, state, orderParametersHistory)
