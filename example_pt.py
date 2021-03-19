import logging
import sys
import time
from decimal import Decimal

import numpy as np

from latticemc.definitions import DefiningParameters, Lattice, LatticeState, OrderParametersHistory
from latticemc.lattice_tools import initialize_partially_ordered
from latticemc.parallel import SimulationRunner
from latticemc.random_quaternion import random_quaternion
from latticemc.updaters import RandomWiggleRateAdjustor

temperatures = np.arange(0.4, 1.2, 0.02)
states = [LatticeState(parameters=DefiningParameters(temperature=round(Decimal(t), 2), lam=round(Decimal(0.3), 2), tau=round(Decimal(1), 1)),
                       lattice=Lattice(8, 8, 8))
          for t in temperatures]
for state in states:
    initialize_partially_ordered(state.lattice, x=random_quaternion(1.0))

order_parameters_history = {state.parameters: OrderParametersHistory() for state in states}

per_state_updaters = [
    # DerivativeWiggleRateAdjustor(howMany=100, how_often=10, since_when=101),
    RandomWiggleRateAdjustor(scale=0.001, how_often=10, since_when=1),
    RandomWiggleRateAdjustor(scale=1.0, reset_value=1.0, how_often=1000, since_when=1000)
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
                              order_parameters_history,
                              cycles=100,
                              report_order_parameters_every=10,
                              report_state_every=10,
                              per_state_updaters=per_state_updaters,
                              parallel_tempering_interval=10)
    runner.start()

    while runner.alive():
        time.sleep(5)
        for state in states:
            mean_energy = order_parameters_history[state.parameters].order_parameters["energy"].mean()
            root.info(f'{state.parameters}: energy={mean_energy}')

    runner.join()
