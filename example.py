from latticemc.definitions import Lattice, LatticeState, OrderParametersHistory, DefiningParameters
from latticemc.latticeTools import initializePartiallyOrdered
from latticemc.randomQuaternion import randomQuaternion
from latticemc.updaters import OrderParametersCalculator, FluctuationsCalculator, DerivativeWiggleRateAdjustor, RandomWiggleRateAdjustor
from latticemc.failsafe import failsafeSaveSimulation
from latticemc import simulationNumba


lattice = Lattice(9, 9, 9)
initializePartiallyOrdered(lattice, x=randomQuaternion(1))
# initializeRandom(lattice)

modelParams = DefiningParameters(temperature=0.9, lam=0.3, tau=1)
state = LatticeState(parameters=modelParams, lattice=lattice)
orderParametersHistory = OrderParametersHistory()

orderParametersCalculator = OrderParametersCalculator(orderParametersHistory, howOften=1, sinceWhen=1, printEvery=50)
fluctuationsCalculator = FluctuationsCalculator(orderParametersHistory, window=100, howOften=50, sinceWhen=100, printEvery=50)
updaters = [
    orderParametersCalculator,
    fluctuationsCalculator,
    DerivativeWiggleRateAdjustor(howMany=100, howOften=10, sinceWhen=101),
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
