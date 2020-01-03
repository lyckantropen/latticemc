from latticemc.definitions import OrderParametersHistory, DefiningParameters, LatticeState, Lattice
from latticemc.updaters import RandomWiggleRateAdjustor
from latticemc.parallel import SimulationRunner
from latticemc.latticeTools import initializePartiallyOrdered
from latticemc.randomQuaternion import randomQuaternion
import numpy as np
import time
import logging
import sys
from decimal import Decimal

temperatures = np.arange(0.4, 1.2, 0.02)
states = [LatticeState(parameters=DefiningParameters(temperature=round(Decimal(t), 2), lam=round(Decimal(0.3), 2), tau=round(Decimal(1), 1)),
                       lattice=Lattice(8, 8, 8))
          for t in temperatures]
for state in states:
    initializePartiallyOrdered(state.lattice, x=randomQuaternion(1.0))

orderParametersHistory = {state.parameters: OrderParametersHistory() for state in states}

perStateUpdaters = [
    # DerivativeWiggleRateAdjustor(howMany=100, howOften=10, sinceWhen=101),
    RandomWiggleRateAdjustor(scale=0.001, howOften=10, sinceWhen=1),
    RandomWiggleRateAdjustor(scale=1.0, resetValue=1.0, howOften=1000, sinceWhen=1000)
]

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s,%(levelname)s: %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

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
            root.info(f'{state.parameters}: energy={orderParametersHistory[state.parameters].orderParameters["energy"].mean()}')

    runner.join()
