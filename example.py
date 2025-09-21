"""
Basic example of a single-temperature lattice Monte Carlo simulation.

This script demonstrates how to set up and run a Monte Carlo simulation
on a 3D lattice of particles with quaternion orientations. The simulation
tracks order parameters and fluctuations over time.
"""

from decimal import Decimal

from latticemc import simulation_numba
from latticemc.definitions import DefiningParameters, Lattice, LatticeState, OrderParametersHistory
from latticemc.failsafe import failsafe_save_simulation
from latticemc.lattice_tools import initialize_partially_ordered
from latticemc.random_quaternion import random_quaternion
from latticemc.updaters import FluctuationsCalculator, OrderParametersCalculator, RandomWiggleRateAdjustor

# Create a 9x9x9 lattice of particles
lattice = Lattice(9, 9, 9)

# Initialize the lattice with partially ordered state
# This gives the system a non-random starting configuration
initialize_partially_ordered(lattice, x=random_quaternion(1))

# Define simulation parameters
# temperature: thermal energy scale (in units where kB = 1)
# lam: coupling strength parameter
# tau: time step parameter
model_params = DefiningParameters(
    temperature=round(Decimal(0.9), 1),
    lam=round(Decimal(0.3), 1),
    tau=round(Decimal(1), 1)
)

# Create the simulation state
state = LatticeState(parameters=model_params, lattice=lattice)

# Initialize data collection for order parameters and fluctuations
order_parameters_history = OrderParametersHistory()

# Set up data collection and system updaters
# Order parameters calculator: computes energy, orientational order parameters every step
order_parameters_calculator = OrderParametersCalculator(
    order_parameters_history, how_often=1, since_when=1, print_every=50
)

# Fluctuations calculator: computes variance in order parameters over a sliding window
fluctuations_calculator = FluctuationsCalculator(
    order_parameters_history, window=100, how_often=50, since_when=100, print_every=50
)

# Set up system updaters for Monte Carlo dynamics
updaters = [
    order_parameters_calculator,    # Track order parameters
    fluctuations_calculator,        # Track fluctuations
    # Adjust wiggle rates to maintain good acceptance rates
    RandomWiggleRateAdjustor(scale=0.001, how_often=10, since_when=1),
    RandomWiggleRateAdjustor(scale=1.0, reset_value=1.0, how_often=1000, since_when=1000),
]

# Run the Monte Carlo simulation
print(f"Starting Monte Carlo simulation at T={model_params.temperature}")
print(f"Lattice size: {lattice.X}×{lattice.Y}×{lattice.Z} = {lattice.X * lattice.Y * lattice.Z} particles")

try:
    # Main simulation loop: 10,000 Monte Carlo steps
    for step in range(10000):
        # Perform one Monte Carlo update step
        simulation_numba.do_lattice_state_update(state)

        # Apply all updaters (data collection, rate adjustments, etc.)
        for u in updaters:
            u.perform(state)

    print("Simulation completed successfully!")
    print(f"Final energy: {order_parameters_history.order_parameters['energy'][-1]:.4f}")
    print(f"Collected {len(order_parameters_history.order_parameters)} order parameter samples")

except Exception as e:
    print(f"Simulation failed with exception: {e}")
    # Save simulation state for debugging
    failsafe_save_simulation(e, state, order_parameters_history)
