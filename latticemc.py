from definitions import Lattice, LatticeState, OrderParametersHistory
from latticeTools import initializePartiallyOrdered, initializeRandom
from randomQuaternion import randomQuaternion
from calculators import OrderParametersCalculator, FluctuationsCalculator, DerivativeWiggleRateAdjustor, RandomWiggleRateAdjustor
from failsafe import failsafeSaveSimulation
import simulationNumba


lattice = Lattice(9, 9, 9)
initializePartiallyOrdered(lattice, x=randomQuaternion(1))
# initializeRandom(lattice)

state = LatticeState(temperature=0.9, lam=0.3, tau=1, lattice=lattice)
orderParametersHistory = OrderParametersHistory()

orderParametersCalculator = OrderParametersCalculator(orderParametersHistory, howOften=1, sinceWhen=1, printEvery=50)
fluctuationsCalculator = FluctuationsCalculator(orderParametersHistory, window=100, howOften=50, sinceWhen=100, printEvery=50)
calculators = [
    orderParametersCalculator,
    fluctuationsCalculator,
    DerivativeWiggleRateAdjustor(howMany=100, howOften=10, sinceWhen=101),
    RandomWiggleRateAdjustor(scale=0.001, howOften=10, sinceWhen=1),
    RandomWiggleRateAdjustor(scale=1.0, resetValue=1.0, howOften=1000, sinceWhen=1000),
]

try:
    for it in range(10000):
        simulationNumba.doLatticeStateUpdate(state)

        for calc in calculators:
            calc.perform(state)

except Exception as e:
    failsafeSaveSimulation(e, state, orderParametersHistory)
