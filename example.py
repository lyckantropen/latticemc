"""
Basic example of a single-temperature lattice Monte Carlo simulation.

This script demonstrates how to set up and run a Monte Carlo simulation
on a 3D lattice of particles with quaternion orientations using the
Simulation class. The simulation tracks order parameters over time.

This example also shows how to use working folders for automatic saving
and recovery of simulation state and results.
"""

import json
import logging
import pathlib
from decimal import Decimal
from typing import List, cast

import numpy as np

from latticemc.definitions import DefiningParameters, Lattice, LatticeState
from latticemc.lattice_tools import initialize_partially_ordered
from latticemc.random_quaternion import random_quaternion
from latticemc.simulation import Simulation
from latticemc.updaters import RandomWiggleRateAdjustor, Updater


def setup_file_logging(working_folder: pathlib.Path, simulation_id: str = "simulation") -> None:
    """
    Set up logging to save all logs to a file in the working folder.

    Args
    ----
    working_folder : pathlib.Path
        Path to the working folder where logs will be saved
    simulation_id : str
        Identifier for this simulation (used in log filename and messages)
    """
    # Create logs directory
    log_dir = working_folder / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file path
    log_file = log_dir / f"{simulation_id}.log"

    # Set up file handler with UTF-8 encoding to handle Greek letters
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Create formatter with simulation ID
    formatter = logging.Formatter(
        f'%(asctime)s - {simulation_id} - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # Get root logger and add our handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Also add console handler if not already present
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Less verbose on console
        console_formatter = logging.Formatter(f'{simulation_id} - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    logging.info(f"Logging set up for {simulation_id}, saving to: {log_file}")


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
# These will be added to the standard order parameters calculators
additional_updaters = [
    # Adjust wiggle rates to maintain good acceptance rates
    RandomWiggleRateAdjustor(scale=0.001, how_often=10, since_when=1),
    RandomWiggleRateAdjustor(scale=1.0, reset_value=1.0, how_often=1000, since_when=1000),
]

# Create and configure the simulation with working folder
# The Simulation class automatically handles:
# - Order parameters calculation (every step)
# - Main simulation loop with proper error handling
# - Automatic progress bar (add progress_bar=None to disable)
# - Automatic saving of simulation state and results when working_folder is provided
working_folder = pathlib.Path("simulation_output")

# Set up logging to save all logs to working folder
# Include simulation parameters in the identifier for easy identification
simulation_id = f"T{model_params.temperature}_lam{model_params.lam}_tau{model_params.tau}"
setup_file_logging(working_folder, simulation_id)

simulation = Simulation(
    initial_state=initial_state,
    cycles=200,  # Number of Monte Carlo steps
    per_state_updaters=cast(List[Updater], additional_updaters),
    working_folder=working_folder,  # Enable automatic saving
    save_interval=100,  # Save every 100 steps
    auto_recover=True  # Attempt to recover from previous runs
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

# Calculate fluctuations from order parameter history
fluctuations = simulation.local_history.calculate_decorrelated_fluctuations()
print("Final fluctuations (calculated from history):")
print(f"  q0 fluctuation: {fluctuations['q0']:.4f}")
print(f"  q2 fluctuation: {fluctuations['q2']:.4f}")

# Demonstrate loading saved data
print("\n" + "=" * 50)
print("DEMONSTRATING DATA LOADING")
print("=" * 50)

# Load and display the saved JSON summary
json_summary_path = working_folder / "summary.json"
if json_summary_path.exists():
    with open(json_summary_path, 'r') as f:
        summary = json.load(f)

    print(f"Loaded JSON summary from: {json_summary_path}")
    print(f"Current step from file: {summary['current_step']}")
    print(f"Total cycles: {summary['total_cycles']}")
    print("Parameters from file:")
    for param, value in summary['parameters'].items():
        print(f"  {param}: {value}")

# Load the complete order parameters history
order_params_path = working_folder / "data" / "order_parameters.npz"
if order_params_path.exists():
    loaded_data = np.load(order_params_path)
    print(f"\nLoaded order parameters from: {order_params_path}")
    print(f"Available arrays: {list(loaded_data.keys())}")

    if 'order_parameters' in loaded_data:
        order_params = loaded_data['order_parameters']
        print(f"Order parameters shape: {order_params.shape}")
        print(f"Energy range: {order_params['energy'].min():.4f} to {order_params['energy'].max():.4f}")

        # Calculate and display some statistics
        print("\nStatistics from saved data:")
        print(f"  Mean energy: {order_params['energy'].mean():.4f}")
        print(f"  Energy std: {order_params['energy'].std():.4f}")
        print(f"  Mean q0: {order_params['q0'].mean():.4f}")
        print(f"  Mean q2: {order_params['q2'].mean():.4f}")

# Load simulation state from JSON summary
json_summary_path = working_folder / "summary.json"
if json_summary_path.exists():
    import json
    with open(json_summary_path, 'r') as f:
        summary = json.load(f)
    print(f"\nLoaded simulation state from: {json_summary_path}")
    print(f"Total cycles: {summary['total_cycles']}")
    print(f"Current step: {summary['current_step']}")
    print(f"Finished: {summary['finished']}")
    print(f"Running time: {summary['running_time_seconds']:.2f} seconds")

print(f"\nAll simulation data saved in: {working_folder.absolute()}")
print("You can examine the files manually or load them in other scripts/notebooks.")
