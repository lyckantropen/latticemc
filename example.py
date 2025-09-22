"""
Basic example of a single-temperature lattice Monte Carlo simulation.

This script demonstrates how to set up and run a Monte Carlo simulation
on a 3D lattice of particles with quaternion orientations using the
Simulation class. The simulation tracks order parameters and fluctuations over time.
"""

from decimal import Decimal

from latticemc.definitions import DefiningParameters, Lattice, LatticeState
from latticemc.lattice_tools import initialize_partially_ordered
from latticemc.random_quaternion import random_quaternion
from latticemc.simulation import Simulation
from latticemc.updaters import RandomWiggleRateAdjustor, Updater
from typing import List, cast

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
initial_state = LatticeState(parameters=model_params, lattice=lattice)

# Set up additional updaters for rate adjustments
# These will be added to the standard order parameters and fluctuations calculators
additional_updaters = [
    # Adjust wiggle rates to maintain good acceptance rates
    RandomWiggleRateAdjustor(scale=0.001, how_often=10, since_when=1),
    RandomWiggleRateAdjustor(scale=1.0, reset_value=1.0, how_often=1000, since_when=1000),
]

# Create and configure the simulation
# The Simulation class automatically handles:
# - Order parameters calculation (every step)
# - Fluctuations calculation (100-step sliding window)
# - Main simulation loop with proper error handling
# - Automatic progress bar (add progress_bar=None to disable)
simulation = Simulation(
    initial_state=initial_state,
    cycles=1000,  # Number of Monte Carlo steps
    fluctuations_window=100,  # Window size for fluctuations calculation
    per_state_updaters=cast(List[Updater], additional_updaters)
    # progress_bar=None  # Uncomment to disable progress bar
)

# Run the Monte Carlo simulation
print(f"Starting Monte Carlo simulation at T={model_params.temperature}")
print(f"Lattice size: {lattice.X}×{lattice.Y}×{lattice.Z} = {lattice.X * lattice.Y * lattice.Z} particles")

# Execute the simulation
simulation.run()

# Access results from the simulation's local history
print("Simulation completed successfully!")
print(f"Final energy: {simulation.local_history.order_parameters['energy'][-1]:.4f}")
print(f"Collected {len(simulation.local_history.order_parameters)} order parameter samples")

# Print some additional statistics
if len(simulation.local_history.order_parameters) > 0:
    print("Final order parameters:")
    print(f"  q0: {simulation.local_history.order_parameters['q0'][-1]:.4f}")
    print(f"  q2: {simulation.local_history.order_parameters['q2'][-1]:.4f}")
    print(f"  p: {simulation.local_history.order_parameters['p'][-1]:.4f}")

if len(simulation.local_history.fluctuations) > 0:
    print("Final fluctuations:")
    print(f"  q0 fluctuation: {simulation.local_history.fluctuations['q0'][-1]:.4f}")
    print(f"  q2 fluctuation: {simulation.local_history.fluctuations['q2'][-1]:.4f}")
