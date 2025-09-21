#!/usr/bin/env python3
"""
Example demonstrating synchronized parallel tempering with barrier synchronization.

This example shows how to use the new replica_exchange_interval parameter
to control when replica exchanges are attempted, and how barrier synchronization
ensures all replicas are synchronized at exchange points.
"""

from decimal import Decimal
from latticemc.definitions import DefiningParameters, LatticeState, Lattice, OrderParametersHistory
from latticemc.lattice_tools import initialize_random
from latticemc.parallel import SimulationRunner


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
    order_parameters_history = {state.parameters: OrderParametersHistory() for state in states}

    print("Setting up synchronized parallel tempering simulation...")
    print("- Replica exchange will be attempted every 100 iterations")
    print("- All replicas will synchronize at barriers before attempting exchange")
    print("- Temperature ladder:", [float(s.parameters.temperature) for s in states])

    # Create simulation runner with synchronized parallel tempering
    runner = SimulationRunner(
        initial_states=states,
        order_parameters_history=order_parameters_history,
        cycles=1000,  # Short run for demonstration
        report_order_parameters_every=200,
        report_fluctuations_every=200,
        report_state_every=200,
        parallel_tempering_interval=100,  # Synchronized exchange every 100 iterations
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


if __name__ == "__main__":
    main()
