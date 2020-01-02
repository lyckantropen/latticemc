from latticemc.definitions import OrderParametersHistory, DefiningParameters, LatticeState, Lattice
from latticemc.updaters import DerivativeWiggleRateAdjustor, RandomWiggleRateAdjustor
from latticemc.parallel import SimulationRunner
import numpy as np
import time

temperatures = np.arange(0.2, 1, 0.1)
states = [LatticeState(parameters=DefiningParameters(temperature=t, lam=0.3, tau=1),
                       lattice=Lattice(8, 8, 8))
          for t in temperatures]
orderParametersHistory = {state.parameters: OrderParametersHistory() for state in states}

perStateUpdaters = [
    DerivativeWiggleRateAdjustor(howMany=100, howOften=10, sinceWhen=101),
    RandomWiggleRateAdjustor(scale=0.001, howOften=10, sinceWhen=1),
    RandomWiggleRateAdjustor(scale=1.0, resetValue=1.0, howOften=1000, sinceWhen=1000)
]

if __name__ == "__main__":
    runner = SimulationRunner(states,
                              orderParametersHistory,
                              cycles=100,
                              reportOrderParametersEvery=10,
                              reportStateEvery=10,
                              perStateUpdaters=perStateUpdaters,
                              parallelTemperingInterval=10)
    runner.start()

    while runner.alive():
        time.sleep(5)
        for state in states:
            print(f'{state.parameters}: energy={orderParametersHistory[state.parameters].orderParameters["energy"].mean()}')

    runner.join()
