from decimal import Decimal

from latticemc import simulation_numba
from latticemc.definitions import DefiningParameters, Lattice, LatticeState, OrderParametersHistory
from latticemc.failsafe import failsafe_save_simulation
from latticemc.lattice_tools import initialize_partially_ordered
from latticemc.random_quaternion import random_quaternion
from latticemc.updaters import DerivativeWiggleRateAdjustor, FluctuationsCalculator, OrderParametersCalculator, RandomWiggleRateAdjustor

lattice = Lattice(9, 9, 9)
initialize_partially_ordered(lattice, x=random_quaternion(1))
# initialize_random(lattice)

model_params = DefiningParameters(temperature=round(Decimal(0.9), 1), lam=round(Decimal(0.3), 1), tau=round(Decimal(1), 1))
state = LatticeState(parameters=model_params, lattice=lattice)
order_parameters_history = OrderParametersHistory()

order_parameters_calculator = OrderParametersCalculator(order_parameters_history, how_often=1, since_when=1, print_every=50)
fluctuations_calculator = FluctuationsCalculator(order_parameters_history, window=100, how_often=50, since_when=100, print_every=50)
updaters = [
    order_parameters_calculator,
    fluctuations_calculator,
    DerivativeWiggleRateAdjustor(order_parameters_history, how_many=100, how_often=10, since_when=101),
    RandomWiggleRateAdjustor(scale=0.001, how_often=10, since_when=1),
    RandomWiggleRateAdjustor(scale=1.0, reset_value=1.0, how_often=1000, since_when=1000),
]

try:
    for _ in range(10000):
        simulation_numba.do_lattice_state_update(state)

        for u in updaters:
            u.perform(state)

except Exception as e:
    failsafe_save_simulation(e, state, order_parameters_history)
