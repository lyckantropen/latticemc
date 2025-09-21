import multiprocessing as mp
from decimal import Decimal
from typing import List

import numpy as np
import pytest

from latticemc.definitions import DefiningParameters, Lattice, LatticeState, OrderParametersHistory
from latticemc.lattice_tools import initialize_random
from latticemc.parallel import ParallelTemperingParameters, SimulationRunner


class TestParallelTempering:
    """Test suite for parallel tempering functionality."""

    def _create_test_states(self, temperatures: List[float], lattice_size: int = 3) -> List[LatticeState]:
        """Create test lattice states with different temperatures."""
        states = []
        for temp in temperatures:
            params = DefiningParameters(
                temperature=Decimal(str(temp)),
                tau=Decimal('0.1'),
                lam=Decimal('1.0')
            )
            lattice = Lattice(X=lattice_size, Y=lattice_size, Z=lattice_size)
            initialize_random(lattice)
            state = LatticeState(parameters=params, lattice=lattice)
            states.append(state)
        return states

    def test_parallel_tempering_decision(self):
        """Test the parallel tempering acceptance decision logic."""
        from latticemc.parallel import SimulationRunner

        # Create mock parameters
        pipe1, pipe2 = mp.Pipe()

        # Test that decision function returns boolean values
        params1 = ParallelTemperingParameters(
            parameters=DefiningParameters(Decimal('1.0'), Decimal('0.1'), Decimal('1.0')),
            energy=100.0,
            pipe=pipe1
        )
        params2 = ParallelTemperingParameters(
            parameters=DefiningParameters(Decimal('2.0'), Decimal('0.1'), Decimal('1.0')),
            energy=80.0,
            pipe=pipe2
        )

        # Test that the function works and returns boolean
        for _ in range(10):
            decision = SimulationRunner._parallel_tempering_decision(params1, params2)
            assert isinstance(decision, bool), "Decision should be boolean"

        # Test with identical parameters (d_b = 0, d_e = 0, so exp(0) = 1, always accept)
        identical_params = ParallelTemperingParameters(
            parameters=DefiningParameters(Decimal('1.5'), Decimal('0.1'), Decimal('1.0')),
            energy=90.0,
            pipe=pipe1
        )

        # With identical parameters, should always accept (exp(0) = 1)
        decision = SimulationRunner._parallel_tempering_decision(identical_params, identical_params)
        assert decision, "Identical parameters should always be accepted (exp(0) = 1)"

    def test_temperature_ladder_creation(self):
        """Test that temperature ladder is correctly created."""
        temperatures = [1.0, 1.5, 2.0, 2.5]
        states = self._create_test_states(temperatures)

        order_parameters_history = {state.parameters: OrderParametersHistory() for state in states}

        runner = SimulationRunner(
            initial_states=states,
            order_parameters_history=order_parameters_history,
            cycles=10  # Very short for testing
        )

        # Check temperature ladder
        assert len(runner._temperatures) == 4
        expected_temps = [Decimal('1.0'), Decimal('1.5'), Decimal('2.0'), Decimal('2.5')]
        assert runner._temperatures == expected_temps

    def test_barrier_creation_with_parallel_tempering(self):
        """Test that barrier is created when parallel tempering is enabled."""
        temperatures = [1.0, 2.0]
        states = self._create_test_states(temperatures)

        order_parameters_history = {state.parameters: OrderParametersHistory() for state in states}

        # Test with parallel tempering enabled
        runner = SimulationRunner(
            initial_states=states,
            order_parameters_history=order_parameters_history,
            cycles=10,
            parallel_tempering_interval=5
        )

        # Barrier should be created
        assert runner._exchange_barrier is not None
        assert runner._exchange_barrier.parties == 2

    def test_no_barrier_without_parallel_tempering(self):
        """Test that no barrier is created when parallel tempering is disabled."""
        states = self._create_test_states([1.0, 2.0])

        order_parameters_history = {state.parameters: OrderParametersHistory() for state in states}

        # Test without parallel tempering
        runner = SimulationRunner(
            initial_states=states,
            order_parameters_history=order_parameters_history,
            cycles=10
            # No parallel_tempering_interval specified
        )

        # No barrier should be created
        assert runner._exchange_barrier is None

    def test_no_barrier_with_single_replica(self):
        """Test that no barrier is created with only one replica."""
        states = self._create_test_states([1.0])  # Only one temperature

        order_parameters_history = {state.parameters: OrderParametersHistory() for state in states}

        runner = SimulationRunner(
            initial_states=states,
            order_parameters_history=order_parameters_history,
            cycles=10,
            parallel_tempering_interval=5
        )

        # No barrier should be created for single replica
        assert runner._exchange_barrier is None

    @pytest.mark.timeout(30)  # Prevent hanging
    def test_short_parallel_tempering_run(self):
        """Test a short parallel tempering simulation run."""
        temperatures = [1.0, 2.0]  # Just two temperatures for simplicity
        states = self._create_test_states(temperatures, lattice_size=2)  # Very small lattice

        order_parameters_history = {state.parameters: OrderParametersHistory() for state in states}

        runner = SimulationRunner(
            initial_states=states,
            order_parameters_history=order_parameters_history,
            cycles=50,  # Short run
            report_order_parameters_every=25,
            report_fluctuations_every=25,
            report_state_every=25,
            parallel_tempering_interval=10  # Attempt exchange every 10 iterations
        )

        # Start simulation
        runner.start()

        # Wait for completion with timeout
        runner.join(timeout=20)

        # Check that simulation completed
        assert not runner.is_alive()

        # Check that everything ran without exceptions
        assert runner.finished_gracefully(), "Simulation did not finish gracefully"

        # Check that some data was collected
        for params, history in order_parameters_history.items():
            # Should have at least some order parameters data
            assert len(history.order_parameters) > 0
            # Should have some reasonable energy values
            energies = history.order_parameters['energy']
            assert len(energies) > 0
            assert all(np.isfinite(energies))

    def test_synchronized_exchange_logic(self):
        """Test the synchronized exchange logic."""
        temperatures = [1.0, 1.5, 2.0]
        states = self._create_test_states(temperatures)

        order_parameters_history = {state.parameters: OrderParametersHistory() for state in states}

        runner = SimulationRunner(
            initial_states=states,
            order_parameters_history=order_parameters_history,
            cycles=10,
            parallel_tempering_interval=5
        )

        # Create mock parallel tempering parameters
        pipes = [mp.Pipe() for _ in range(3)]
        pt_ready = []
        for i, temp in enumerate(temperatures):
            pt_param = ParallelTemperingParameters(
                parameters=DefiningParameters(Decimal(str(temp)), Decimal('0.1'), Decimal('1.0')),
                energy=100.0 - i * 10,  # Decreasing energy with increasing temperature
                pipe=pipes[i][1]
            )
            # pt_ready now contains tuples of (simulation_index, ParallelTemperingParameters)
            pt_ready.append((i, pt_param))

        # Test synchronized exchange with all replicas ready
        runner._do_synchronized_exchange(pt_ready)

        # Check that all pipes received parameters (exchange decisions were made)
        for i, (our_pipe, _) in enumerate(pipes):
            assert our_pipe.poll(), f"Pipe {i} should have received parameters"
            received_params = our_pipe.recv()
            assert received_params is not None
            assert hasattr(received_params, 'temperature')

    @pytest.mark.timeout(45)  # Prevent hanging if deadlock occurs
    def test_barrier_deadlock_prevention(self):
        """Test that barriers don't cause deadlocks when simulations end at different times."""
        # Create test states with multiple replicas to ensure barrier usage
        temperatures = [1.0, 2.0, 3.0]
        states = self._create_test_states(temperatures, lattice_size=2)  # Small lattice for speed

        order_parameters_history = {state.parameters: OrderParametersHistory() for state in states}

        # Create runner with parallel tempering and frequent exchanges to increase chance of edge case
        runner = SimulationRunner(
            initial_states=states,
            order_parameters_history=order_parameters_history,
            cycles=100,  # Short simulation to finish quickly
            report_order_parameters_every=50,
            report_fluctuations_every=50,
            report_state_every=50,
            parallel_tempering_interval=10  # Frequent exchanges to test barrier edge cases
        )

        # Verify barrier was created (prerequisite for this test)
        assert runner._exchange_barrier is not None
        assert runner._exchange_barrier.parties == 3

        # Start simulation
        runner.start()

        # Wait for completion with timeout - if deadlock occurs, timeout will trigger
        runner.join(timeout=30)

        # Check that everything ran without exceptions
        assert runner.finished_gracefully(), "Simulation did not finish gracefully"

        # Verify simulation completed successfully without hanging
        assert not runner.is_alive(), "Simulation should have completed without deadlock"

    def test_process_crash_exception_propagation(self):
        """Test that SimulationRunner raises exception when a process crashes."""
        states = self._create_test_states([1.0, 2.0], lattice_size=2)
        order_parameters_history = {state.parameters: OrderParametersHistory() for state in states}

        runner = SimulationRunner(
            initial_states=states,
            order_parameters_history=order_parameters_history,
            cycles=1000,  # Longer simulation to allow time for crash detection
            parallel_tempering_interval=10
        )

        # Start runner
        runner.start()

        # Simulate a process crash by manually terminating one process
        import time
        time.sleep(0.5)  # Let processes start and send initial pings
        if runner.simulations:
            runner.simulations[0].terminate()  # Force kill one process
            runner.simulations[0].join(timeout=1)  # Wait for it to die

        # Wait for runner to detect the crash - should happen within the thread
        runner.join(timeout=15)  # Give enough time for ping timeout detection

        # Check if runner detected the crash (it should have stopped due to exception)
        assert not runner.alive(), "Runner should have stopped due to process crash"
        assert not runner.finished_gracefully(), "Runner should not have finished gracefully after process crash"

    def test_ping_mechanism_during_normal_operation(self):
        """Test that ping messages are sent and received during normal operation."""
        states = self._create_test_states([1.0, 2.0], lattice_size=2)
        order_parameters_history = {state.parameters: OrderParametersHistory() for state in states}

        runner = SimulationRunner(
            initial_states=states,
            order_parameters_history=order_parameters_history,
            cycles=2000,  # Enough cycles to trigger ping messages
            parallel_tempering_interval=50
        )

        # Start runner
        runner.start()
        runner.join(timeout=30)

        # Check that everything ran without exceptions
        assert runner.finished_gracefully(), "Simulation did not finish gracefully"

        # Verify simulation completed successfully
        assert not runner.is_alive(), "Runner should have completed successfully"

        # Verify that ping messages were received (ping tracking dict should have entries)
        assert len(runner._last_ping_time) > 0, "Should have received ping messages from processes"
        assert len(runner._last_ping_time) <= len(states), "Should not have more pings than processes"


if __name__ == "__main__":
    pytest.main([__file__])
