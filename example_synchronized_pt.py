#!/usr/bin/env python3
"""
Example demonstrating synchronized parallel tempering with barrier synchronization.

This example shows how to use the new replica_exchange_interval parameter
to control when replica exchanges are attempted, and how barrier synchronization
ensures all replicas are synchronized at exchange points.

This example also demonstrates working folder functionality for saving
and loading parallel tempering simulation data.
"""

import logging
import json
import pathlib
from decimal import Decimal

import numpy as np

from latticemc.definitions import DefiningParameters, Lattice, LatticeState, OrderParametersHistory
from latticemc.lattice_tools import initialize_random
from latticemc.parallel import SimulationRunner


def setup_parallel_tempering_logging(working_folder: pathlib.Path) -> None:
    """
    Set up logging for parallel tempering main process.

    Individual simulation processes will create their own log files
    in their respective process folders.

    Args
    ----
    working_folder : pathlib.Path
        Path to the working folder where main process logs will be saved
    """
    # Create logs directory for main process
    log_dir = working_folder / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file path for main coordination process
    log_file = log_dir / "parallel_tempering_main.log"

    # Set up file handler with UTF-8 encoding to handle Greek letters
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Create formatter with process identification
    formatter = logging.Formatter(
        '%(asctime)s - PT_MAIN - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # Get root logger and add our handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Also add console handler if not already present
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('PT_MAIN - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    logging.info(f"Main parallel tempering logging set up, saving to: {log_file}")
    logging.info("Individual processes will create their own log files in process_XXX/logs/ folders")


def create_example_states():
    """Create example lattice states with different temperatures for parallel tempering."""
    temperatures = [Decimal('1.0'), Decimal('1.5'), Decimal('2.0'), Decimal('2.5')]
    states = []

    for temp in temperatures:
        # Create defining parameters
        params = DefiningParameters(temperature=temp, tau=Decimal('0.1'), lam=Decimal('1.0'))

        # Create lattice
        lattice = Lattice(X=4, Y=4, Z=4)  # Small lattice for demonstration
        initialize_random(lattice)

        # Create lattice state
        state = LatticeState(parameters=params, lattice=lattice)
        states.append(state)

    return states


def main():
    """Demonstrate synchronized parallel tempering."""
    print("Creating example lattice states...")
    states = create_example_states()

    # Create order parameters history dictionary
    order_parameters_history = {state.parameters: OrderParametersHistory(state.lattice.size) for state in states}

    print("Setting up synchronized parallel tempering simulation...")
    print("- Replica exchange will be attempted every 100 iterations")
    print("- All replicas will synchronize at barriers before attempting exchange")
    print("- Temperature ladder:", [float(s.parameters.temperature) for s in states])

    # Set up working folder for parallel tempering data saving
    working_folder = pathlib.Path("parallel_tempering_output")

    # Set up logging for main parallel tempering process
    # Each simulation process will create its own log files in their process folders
    setup_parallel_tempering_logging(working_folder)

    # Create simulation runner with synchronized parallel tempering and working folder
    runner = SimulationRunner(
        initial_states=states,
        order_parameters_history=order_parameters_history,
        cycles=1000,  # Short run for demonstration
        report_order_parameters_every=200,
        report_fluctuations_every=200,
        report_state_every=200,
        parallel_tempering_interval=100,  # Synchronized exchange every 100 iterations
        working_folder=str(working_folder),  # Enable automatic saving for each process
        save_interval=500,  # Save every 500 steps
        auto_recover=True  # Enable recovery from saved state
    )

    print("\nStarting simulation with synchronized replica exchange...")
    runner.start()
    runner.join()

    print("\nSimulation completed!")
    print("Order parameters histories:")
    for params, history in order_parameters_history.items():
        temp = float(params.temperature)
        n_points = len(history.order_parameters)
        print(f"  Temperature {temp}: {n_points} data points")

        # Show some basic statistics for each temperature
        if n_points > 0:
            print(f"    Final energy: {history.order_parameters['energy'][-1]:.4f}")
            print(f"    Mean energy: {history.order_parameters['energy'].mean():.4f}")

    # Demonstrate loading saved parallel tempering data
    print("\n" + "=" * 50)
    print("DEMONSTRATING PARALLEL TEMPERING DATA LOADING")
    print("=" * 50)

    # Load saved data for each temperature process
    if working_folder.exists():
        print(f"Working folder: {working_folder.absolute()}")

        # List all process folders
        process_folders = [f for f in working_folder.iterdir() if f.is_dir() and f.name.startswith('process_')]
        process_folders.sort()

        print(f"\nFound {len(process_folders)} process folders:")
        for i, process_folder in enumerate(process_folders):
            print(f"  {process_folder.name}")

            # Determine which temperature this process corresponds to
            if i < len(states):
                temp = float(states[i].parameters.temperature)
                print(f"    Temperature: {temp:.1f}")

                # Load JSON summary
                json_file = process_folder / "summary.json"
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        summary = json.load(f)
                    print(f"    Steps completed: {summary['current_step']}")
                    if 'latest_order_parameters' in summary and 'energy' in summary['latest_order_parameters']:
                        print(f"    Final energy: {summary['latest_order_parameters']['energy']:.4f}")

                # Load order parameters
                npz_file = process_folder / "data" / "order_parameters.npz"
                if npz_file.exists():
                    data = np.load(npz_file)
                    if 'order_parameters' in data:
                        ops = data['order_parameters']
                        print(f"    Data points: {len(ops)}")
                        print(f"    Energy range: {ops['energy'].min():.4f} to {ops['energy'].max():.4f}")
                        print(f"    Mean q0: {ops['q0'].mean():.4f} ± {ops['q0'].std():.4f}")
                else:
                    print("    ❌ No order parameters file found")
    else:
        print("❌ Working folder not found")

    print(f"\nAll parallel tempering data saved in: {working_folder.absolute()}")
    print("Each temperature process has its own subfolder with complete simulation data.")
    print("This enables:")
    print("  • Individual process recovery and continuation")
    print("  • Detailed analysis of each temperature replica")
    print("  • Post-processing of parallel tempering results")


if __name__ == "__main__":
    main()
