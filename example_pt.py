from latticemc.definitions import OrderParametersHistory, DefiningParameters
from latticemc.updaters import DerivativeWiggleRateAdjustor, RandomWiggleRateAdjustor
from latticemc.parallel import SimulationRunner
import numpy as np
import time

temperatures = np.arange(0.2, 1, 0.1)
parameters = [DefiningParameters(temperature=t, lam=0.3, tau=1) for t in temperatures]
orderParametersHistory = {p: OrderParametersHistory() for p in parameters}

perStateUpdaters = [
    DerivativeWiggleRateAdjustor(howMany=100, howOften=10, sinceWhen=101),
    RandomWiggleRateAdjustor(scale=0.001, howOften=10, sinceWhen=1),
    RandomWiggleRateAdjustor(scale=1.0, resetValue=1.0, howOften=1000, sinceWhen=1000)
]

if __name__ == "__main__":
    runner = SimulationRunner(parameters,
                              orderParametersHistory,
                              latticeSize=(8, 8, 8),
                              cycles=10000,
                              reportOrderParametersEvery=100,
                              perStateUpdaters=perStateUpdaters,
                              parallelTemperingInterval=100)
    runner.start()

    while runner.alive():
        time.sleep(5)
        for p in parameters:
            print(f'{p}: energy={orderParametersHistory[p].orderParameters["energy"].mean()}')

    runner.join()
