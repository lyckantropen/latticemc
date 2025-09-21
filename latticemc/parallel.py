"""Parallel processing and threading support for lattice simulations."""

import logging
import multiprocessing as mp
import threading
import time
from collections import defaultdict, namedtuple
from ctypes import c_double
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, cast

import numpy as np

from . import simulation_numba
from .definitions import DefiningParameters, LatticeState, OrderParametersHistory
from .failsafe import failsafe_save_simulation
from .tensorboard_logger import TensorBoardLogger
from .updaters import CallbackUpdater, FluctuationsCalculator, OrderParametersCalculator, Updater

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """
    The type of message that `SimulationProcess` can send to `SimulationRunner`.

    All messages have the format of a tuple: `(MessageType, self.index, payload)`
    """

    OrderParameters = 1
    Fluctuations = 2
    State = 3
    ParallelTemperingSignUp = 4
    Error = 5
    Finished = 6
    Ping = 7


ParallelTemperingParameters = namedtuple('ParallelTemperingParameters', ['parameters', 'energy', 'pipe'])


class SimulationProcess(mp.Process):
    """
    Process representing one simulation over a lattice of particles.

    When parallel tempering is enabled, does not necessarily mean a
    particular configuration of parameters. Otherwise the parameters
    are constant.
    """

    def __init__(self,
                 index: int,
                 queue: mp.Queue,
                 initial_state: LatticeState,
                 cycles: int,
                 report_order_parameters_every: int = 1000,
                 report_fluctuations_every: int = 1000,
                 report_state_every: int = 1000,
                 fluctuations_window: int = 1000,
                 per_state_updaters: List[Updater] = [],
                 parallel_tempering_interval: Optional[int] = None,
                 exchange_barrier=None
                 ) -> None:
        super().__init__()
        self.index = index
        self.queue = queue
        self.state = initial_state
        self.cycles = cycles
        self.report_order_parameters_every = report_order_parameters_every
        self.report_fluctuations_every = report_fluctuations_every
        self.report_state_every = report_state_every
        self.per_state_updaters = per_state_updaters
        self.fluctuations_window = fluctuations_window
        self.parallel_tempering_interval = parallel_tempering_interval
        self.exchange_barrier = exchange_barrier

        self.running = mp.Value('i', 1)
        self.temperature = mp.Value(c_double, float(self.state.parameters.temperature))

        # how many data points truly belong to the present configuration
        # is important when parallel tempering is enabled
        self._relevant_history_length = 0
        self.local_history = OrderParametersHistory()

    def run(self) -> None:
        """Execute the simulation process."""
        order_parameters_broadcaster = CallbackUpdater(
            callback=lambda _: self._broadcast_order_parameters(),
            how_often=self.report_order_parameters_every,
            since_when=self.report_order_parameters_every
        )
        fluctuations_broadcaster = CallbackUpdater(
            callback=lambda _: self._broadcast_fluctuations(),
            how_often=self.report_fluctuations_every,
            since_when=self.fluctuations_window
        )
        state_broadcaster = CallbackUpdater(
            callback=lambda _: self._broadcast_state(),
            how_often=self.report_state_every,
            since_when=self.report_state_every
        )
        ping_updater = CallbackUpdater(
            callback=lambda _: self._send_ping(),
            how_often=5,  # Send ping every 5 iterations
            since_when=0
        )

        order_parameters_calculator = OrderParametersCalculator(self.local_history, how_often=1, since_when=0)
        fluctuations_calculator = FluctuationsCalculator(self.local_history, window=self.fluctuations_window, how_often=1, since_when=self.fluctuations_window)
        per_state_updaters = [
            order_parameters_calculator,
            fluctuations_calculator,
            *self.per_state_updaters,
            order_parameters_broadcaster,
            fluctuations_broadcaster,
            state_broadcaster,
            ping_updater
        ]

        if self.parallel_tempering_interval is not None and self.exchange_barrier is not None:
            parallel_tempering_updater = CallbackUpdater(
                callback=lambda _: self._parallel_tempering(),
                how_often=self.parallel_tempering_interval,
                since_when=self.parallel_tempering_interval
            )
            per_state_updaters.append(parallel_tempering_updater)

        # MAIN SIMULATION LOOP
        try:
            for _ in range(self.cycles):
                simulation_numba.do_lattice_state_update(self.state)
                self._relevant_history_length += 1
                for u in per_state_updaters:
                    u.perform(self.state)
        except Exception as e:
            self.queue.put((MessageType.Error, self.index, (self.state.parameters, e)))
            failsafe_save_simulation(e, self.state, self.local_history)

        # Mark as finished - this will prevent future barrier waits
        self.running.value = 0

        self.queue.put((MessageType.State, self.index, self.state))
        self.queue.put((MessageType.Finished, self.index, self.state.parameters))

    def _broadcast_order_parameters(self) -> None:
        """Publish at most `self._relevant_history_length` order parameters from history to the governing thread."""
        self.queue.put((MessageType.OrderParameters, self.index,
                        (self.state.parameters,
                         self.local_history.order_parameters[-min(self._relevant_history_length, self.report_order_parameters_every):])))

    def _broadcast_fluctuations(self) -> None:
        """Publish at most `self._relevant_history_length` fluctuation values from history to the governing thread."""
        self.queue.put((MessageType.Fluctuations, self.index,
                        (self.state.parameters,
                         self.local_history.fluctuations[-min(self._relevant_history_length, self.report_fluctuations_every):])))

    def _broadcast_state(self):
        """Publish the current Lattice State to the governing thread."""
        self.queue.put((MessageType.State, self.index, self.state))

    def _send_ping(self) -> None:
        """Send a ping message to indicate the process is alive and healthy."""
        self.queue.put((MessageType.Ping, self.index, self.state.iterations))

    def _parallel_tempering(self) -> None:
        """Perform synchronized parallel tempering update using barrier synchronization."""
        # Check if we're still supposed to be running - if not, skip parallel tempering
        if not self.running.value:
            logger.debug(f'SimulationProcess[{self.index}, {self.state.parameters}]: Skipping parallel tempering - simulation ending')
            return

        # Wait at barrier to synchronize all replicas before attempting exchange
        if self.exchange_barrier is not None:
            logger.debug(f'SimulationProcess[{self.index}, {self.state.parameters}]: Waiting at exchange barrier')
            try:
                self.exchange_barrier.wait(timeout=30)  # Add timeout to prevent indefinite hanging
            except Exception as e:
                # Handle barrier broken, timeout, or other barrier-related exceptions
                logger.debug(f'SimulationProcess[{self.index}, {self.state.parameters}]: Barrier exception ({e}), skipping parallel tempering')
                return

        # energy needs to be scaled by number of particles (total system energy not per-particle)
        # the energy stored in order parameters is a lattice average
        energy = self.local_history.order_parameters['energy'][-1] * cast(np.ndarray, self.state.lattice.particles).size
        our, theirs = mp.Pipe()
        self.queue.put((MessageType.ParallelTemperingSignUp, self.index, ParallelTemperingParameters(
            parameters=self.state.parameters, energy=energy, pipe=theirs)))

        # wait for decision in governing thread
        if not our.poll(30):
            logger.warning(f'SimulationProcess[{self.index}, {self.state.parameters}]: No parallel tempering data to exchange')
            return
        parameters = our.recv()

        logger.debug(f'SimulationProcess[{self.index}, {self.state.parameters}]: Received parameters for exchange: {parameters}')

        if parameters != self.state.parameters:
            # broadcast what we can
            self._broadcast_order_parameters()
            self._broadcast_fluctuations()
            self._broadcast_state()

            # parameter change
            with self.temperature.get_lock():
                self.state.parameters = parameters
                self.temperature.value = float(parameters.temperature)

            # reset the number of results that can be safely broadcasted as coming from this configuration
            self._relevant_history_length = 0


class SimulationRunner(threading.Thread):
    """Thread that manages multiple simulation processes."""

    def __init__(self,
                 initial_states: List[LatticeState],
                 order_parameters_history: Dict[DefiningParameters, OrderParametersHistory],
                 cycles: int,
                 *args,
                 tensorboard_log_dir: Optional[str] = None,
                 parallel_tempering_interval: Optional[int] = None,
                 **kwargs) -> None:
        threading.Thread.__init__(self)

        self.cycles = cycles
        self.states = initial_states
        self.order_parameters_history = order_parameters_history
        self.args = args
        self.kwargs = kwargs
        self.simulations: List[SimulationProcess] = []
        self.progress_meters: Dict[int, int] = defaultdict(lambda: 0)

        # set to False when all processes have started running
        self._starting = True

        # Track process errors and failures
        self._process_errors: List[Exception] = []
        self._crashed_processes: List[int] = []

        # Track process health via ping messages
        self._last_ping_time: Dict[int, float] = {}
        self._ping_timeout = 120.0  # seconds without ping before considering process dead

        # for parallel tempering
        self._temperatures: List[Decimal] = [state.parameters.temperature for state in self.states]
        self._exchange_counter = 0  # Global counter for replica exchange rounds

        # barrier for synchronization during replica exchange
        # Create barrier if parallel tempering is enabled and there are multiple replicas
        self.parallel_tempering_enabled = parallel_tempering_interval is not None
        self.kwargs['parallel_tempering_interval'] = parallel_tempering_interval
        self._exchange_barrier = None
        if self.parallel_tempering_enabled and len(self.states) > 1:
            self._exchange_barrier = mp.Barrier(len(self.states))

        # TensorBoard logging setup - always enabled
        self.tb_logger: Optional[TensorBoardLogger]
        try:
            self.tb_logger = TensorBoardLogger(log_dir=tensorboard_log_dir)
            logger.info(f"TensorBoard logging enabled, writing to {self.tb_logger.log_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize TensorBoard logging: {e}")
            self.tb_logger = None

        # Start time for logging
        self._start_time: Optional[float] = None

    def run(self) -> None:
        """Execute all simulation processes."""
        q: mp.Queue = mp.Queue()

        # Log all temperature assignments to TensorBoard
        if self.tb_logger:
            self.tb_logger.log_simulation_temperature_assignments(self.states)

        self._start_time = time.time()
        last_logged_time = self._start_time

        # create and start all simulations
        self.simulations = []
        for i, state in enumerate(self.states):
            # Prepare kwargs with barrier
            sim_kwargs = dict(self.kwargs)
            sim_kwargs['exchange_barrier'] = self._exchange_barrier
            sim_kwargs['cycles'] = self.cycles
            sim = SimulationProcess(i, q, state, *self.args, **sim_kwargs)
            sim.start()
            self.simulations.append(sim)

            # Log the temperature assignment for each simulation index
            logger.info(f"SimulationProcess[{i}]: Running at temperature T={float(state.parameters.temperature):.4f}")

        self._starting = False

        # main message processing loop
        pt_ready: List[Tuple[int, ParallelTemperingParameters]] = []
        while self.alive():
            while not q.empty():
                message_type, index, msg = q.get()

                logger.debug(f'SimulationRunner: Received {message_type}, index={index}')

                if message_type == MessageType.OrderParameters:
                    parameters, op = msg
                    self.order_parameters_history[parameters].order_parameters = np.append(self.order_parameters_history[parameters].order_parameters, op)
                    # Log order parameters immediately when received
                    self._log_order_parameters_to_tensorboard(parameters, op)
                if message_type == MessageType.Fluctuations:
                    parameters, fl = msg
                    self.order_parameters_history[parameters].fluctuations = np.append(self.order_parameters_history[parameters].fluctuations, fl)
                    # Log fluctuations immediately when received
                    self._log_fluctuations_to_tensorboard(parameters, fl)
                if message_type == MessageType.State:
                    # update the state
                    msg = cast(LatticeState, msg)
                    state = [state for state in self.states if state.parameters == msg.parameters][0]
                    state.update_from(msg)
                    # Log energy and acceptance rates immediately when state is received
                    self._log_state_to_tensorboard(state)
                if message_type == MessageType.ParallelTemperingSignUp:
                    pt_parameters = msg
                    pt_ready.append((index, pt_parameters))
                if message_type == MessageType.Error:
                    parameters, exception = msg
                    logger.error(f'SimulationProcess[{index},{parameters}]: Failed with exception "{exception}"')
                    self._process_errors.append(exception)
                    self._crashed_processes.append(index)
                if message_type == MessageType.Finished:
                    parameters = msg
                    logger.info(f'SimulationProcess[{index},{parameters}]: Finished succesfully')
                if message_type == MessageType.Ping:
                    iterations = msg
                    self._last_ping_time[index] = time.time()
                    self.progress_meters[index] = iterations
                    logger.debug(f'SimulationProcess[{index}]: Ping received at iteration {iterations}')

            # Check for crashed processes and raise exception if any are found
            try:
                self._check_for_crashed_processes()
            except RuntimeError as e:
                logger.error(f'SimulationRunner: Stopping due to error: {e}')
                self.stop()
                raise e

            # Perform parallel tempering if enabled and there are replicas ready
            if self.parallel_tempering_enabled:
                self._do_parallel_tempering(pt_ready)

            # Log simulation progress every 10 seconds
            if self.tb_logger:
                current_time = time.time()
                if current_time - last_logged_time >= 10.0:
                    self.log_temperature_plots_to_tensorboard()
                    last_logged_time = current_time

        # Close TensorBoard logger
        if self.tb_logger:
            self.tb_logger.close()

        # Ensure all processes have finished
        [sim.join() for sim in self.simulations]  # type: ignore

        # Log final temperature plots
        if self.tb_logger:
            self.log_temperature_plots_to_tensorboard(recent_points=self.cycles)

    def _check_for_crashed_processes(self) -> None:
        """Check for crashed processes using ping messages and process status."""
        current_time = time.time()

        # First check for explicit error messages
        if self._process_errors:
            self.stop()
            error_summary = f"Simulation failed: {len(self._crashed_processes)} process(es) crashed"
            error_details = []
            for i, error in enumerate(self._process_errors):
                process_index = self._crashed_processes[i] if i < len(self._crashed_processes) else "unknown"
                error_details.append(f"Process {process_index}: {error}")

            full_error_msg = f"{error_summary}\n" + "\n".join(error_details)
            logger.error(full_error_msg)
            raise RuntimeError(full_error_msg)

        # Check for processes that haven't pinged recently (ping-based health check)
        unresponsive_processes = []
        dead_processes = []

        for i, sim in enumerate(self.simulations):
            if i in self._crashed_processes:
                continue  # Already known to be crashed

            # Check if process is dead
            if not sim.is_alive():
                # Check if process died unexpectedly (non-zero exit code or None means abnormal termination)
                if sim.exitcode != 0:  # Process died unexpectedly (crashed, killed, etc.)
                    dead_processes.append(i)
                continue  # Process finished normally with exit code 0

            # Check ping timeout for running processes
            if i in self._last_ping_time:
                time_since_ping = current_time - self._last_ping_time[i]
                if time_since_ping > self._ping_timeout:
                    unresponsive_processes.append(i)
            # If we haven't received any ping yet, only worry if process has been running for a while
            elif not self._starting and current_time - (self._start_time or current_time) > self._ping_timeout:
                unresponsive_processes.append(i)

        if dead_processes or unresponsive_processes:
            self.stop()
            error_parts = []
            if dead_processes:
                error_parts.append(f"Process(es) {dead_processes} died unexpectedly")
            if unresponsive_processes:
                error_parts.append(f"Process(es) {unresponsive_processes} stopped responding (no ping for >{self._ping_timeout}s)")

            error_msg = f"Simulation failed: {'; '.join(error_parts)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _do_parallel_tempering(self, pt_ready: List[tuple[int, ParallelTemperingParameters]]):
        """Manage synchronized replica exchange between temperature configurations."""
        # With barrier synchronization, all replicas should be ready at the same time
        if len(pt_ready) == len(self.states):
            self._do_synchronized_exchange(pt_ready)
        # else: keep waiting for all replicas (don't log to avoid performance impact)

    def _do_synchronized_exchange(self, pt_ready: List[Tuple[int, ParallelTemperingParameters]]):
        """Perform synchronized replica exchange with all replicas ready."""
        # Sort by temperature for systematic exchange attempts
        pt_ready.sort(key=lambda entry: entry[1].parameters.temperature)

        # Track exchange statistics
        total_attempts = 0
        total_accepted = 0

        # Proper replica exchange using even-odd scheme for better ergodicity
        # This alternates between even pairs (0,1), (2,3)... and odd pairs (1,2), (3,4)...
        # ensuring all replicas can mix efficiently without systematic biases

        # Determine whether to use even or odd pairs (alternates each exchange round)
        # Use global exchange counter to alternate between even and odd pairs
        use_even_pairs = self._exchange_counter % 2 == 0

        # Create appropriate pairs based on even/odd scheme
        if use_even_pairs:
            # Even pairs: (0,1), (2,3), (4,5), ...
            pairs = [(i, i + 1) for i in range(0, len(pt_ready) - 1, 2)]
            logger.debug(f'SimulationRunner: Exchange round {self._exchange_counter} - using even pairs: {pairs}')
        else:
            # Odd pairs: (1,2), (3,4), (5,6), ...
            pairs = [(i, i + 1) for i in range(1, len(pt_ready) - 1, 2)]
            logger.debug(f'SimulationRunner: Exchange round {self._exchange_counter} - using odd pairs: {pairs}')

        # Attempt all pairs simultaneously (no conflicts since pairs don't overlap)
        for i, j in pairs:
            total_attempts += 1
            idx1, p1 = pt_ready[i]
            idx2, p2 = pt_ready[j]
            accepted = self._parallel_tempering_decision(p1, p2)

            if accepted:
                # Exchange parameters - simulation idx1 gets p2's temp, simulation idx2 gets p1's temp
                total_accepted += 1
                p1.pipe.send(p2.parameters)
                p2.pipe.send(p1.parameters)
                # Update pt_ready to reflect the post-exchange temperatures
                pt_ready[i] = (idx1, ParallelTemperingParameters(p2.parameters, p1.energy, p1.pipe))
                pt_ready[j] = (idx2, ParallelTemperingParameters(p1.parameters, p2.energy, p2.pipe))
                logger.debug(f'SimulationRunner: Exchanged {p1.parameters.temperature} (sim {idx1}) and {p2.parameters.temperature} (sim {idx2})')
            else:
                # No exchange - simulations keep their original temperatures
                p1.pipe.send(p1.parameters)
                p2.pipe.send(p2.parameters)
                logger.debug(f'SimulationRunner: Did not exchange {p1.parameters.temperature} (sim {idx1}) and {p2.parameters.temperature} (sim {idx2})')

        # Handle any remaining replicas that weren't paired in the even-odd scheme
        # Create set of indices that were included in pairs
        paired_indices = set()
        for i, j in pairs:
            paired_indices.update([i, j])

        # Send original parameters to unpaired replicas
        for i, (idx, pt_param) in enumerate(pt_ready):
            if i not in paired_indices:
                pt_param.pipe.send(pt_param.parameters)
                logger.debug(f'SimulationRunner: No exchange for unpaired replica {i} (sim {idx}, T={pt_param.parameters.temperature})')

        # Log overall statistics and current temperatures to TensorBoard
        if self.tb_logger:
            # Log current temperature for each process after exchanges using the post-exchange assignments
            self._log_current_temperatures_after_exchange(pt_ready, self._exchange_counter)

        # Increment the global exchange counter after completing this exchange round
        self._exchange_counter += 1

        pt_ready.clear()

    @staticmethod
    def _parallel_tempering_decision(p1: ParallelTemperingParameters, p2: ParallelTemperingParameters) -> bool:
        """
        Decide whether to exchange replicas between two temperatures using Metropolis criterion.

        The acceptance probability is exp(ΔE x Δβ) where:
        - ΔE = E₁ - E₂ (energy difference)
        - Δβ = 1/T₁ - 1/T₂ (inverse temperature difference)

        Automatically accepts exchanges where ΔE x Δβ > 0 (energetically favorable).
        """
        t1, e1, _ = float(p1.parameters.temperature), p1.energy, p1.pipe
        t2, e2, _ = float(p2.parameters.temperature), p2.energy, p2.pipe
        d_b = 1 / t1 - 1 / t2  # Δβ = β₁ - β₂
        d_e = e1 - e2          # ΔE = E₁ - E₂

        # Accept if energetically favorable or passes Metropolis test
        return d_b * d_e > 0 or np.random.random() < np.exp(d_b * d_e)

    def stop(self) -> None:
        """Terminate all simulation processes."""
        [sim.terminate() for sim in self.simulations]  # type: ignore
        [sim.join() for sim in self.simulations if sim.is_alive()]  # type: ignore

    def alive(self) -> bool:
        """Check if any simulation processes are still running."""
        return self._starting or bool([sim for sim in self.simulations if sim.is_alive()])

    def finished_gracefully(self) -> bool:
        """
        Check whether all simulation processes finished gracefully.

        Prints detailed information about any processes that did not finish gracefully.

        Returns
        -------
        bool
            True if all processes have finished with exit code 0 (normal termination),
            False if any processes are still running, crashed, or were terminated.
        """
        if self._starting:
            print("Simulation is still starting up")
            return False

        if self.alive():
            alive_processes = [i for i, sim in enumerate(self.simulations) if sim.is_alive()]
            print(f"Simulation not finished: {len(alive_processes)} process(es) still running: {alive_processes}")
            return False

        # Check if all processes finished with exit code 0
        failed_processes = []
        for i, sim in enumerate(self.simulations):
            if sim.exitcode != 0:
                failed_processes.append((i, sim.exitcode, sim.pid))

        if failed_processes:
            print("Simulation did not finish gracefully:")
            for process_idx, exitcode, pid in failed_processes:
                if exitcode == -15:  # SIGTERM
                    reason = "terminated (SIGTERM - likely killed due to resource limits or manual termination)"
                elif exitcode == -9:   # SIGKILL
                    reason = "killed (SIGKILL - forcibly terminated)"
                elif exitcode == -11:  # SIGSEGV
                    reason = "crashed (SIGSEGV - segmentation fault)"
                elif cast(int, exitcode) < 0:
                    reason = f"killed by signal {-cast(int, exitcode)}"
                elif cast(int, exitcode) > 0:
                    reason = f"exited with error code {exitcode}"
                else:
                    reason = f"unknown exit condition (exitcode={exitcode})"

                print(f"  Process {process_idx} (PID {pid}): {reason}")
            return False

        print("All simulation processes finished gracefully")
        return True

    def get_process_status(self) -> Dict[int, Dict[str, Dict]]:
        """Get detailed status information for all simulation processes."""
        status: Dict[int, Dict] = {}
        for i, sim in enumerate(self.simulations):
            status[i] = {
                'alive': sim.is_alive(),
                'exitcode': sim.exitcode,
                'pid': sim.pid,
                'graceful': sim.exitcode == 0 if sim.exitcode is not None else None
            }
        return status

    def _log_order_parameters_to_tensorboard(self, parameters: DefiningParameters, op: np.ndarray) -> None:
        """Log order parameters immediately when received."""
        if not self.tb_logger:
            return

        try:
            # Use the length of order parameters history as step
            step = len(self.order_parameters_history[parameters].order_parameters) if parameters in self.order_parameters_history else None

            # Log individual order parameter values for this temperature
            for field in self.tb_logger.order_parameter_fields:
                if len(op) > 0:
                    value = float(op[-1][field])  # Get the most recent value
                    self.tb_logger.log_temperature_scalar_auto('order_parameters', field, parameters, value, step)

        except Exception as e:
            logger.error(f"Error logging order parameters to TensorBoard: {e}")

    def _log_fluctuations_to_tensorboard(self, parameters: DefiningParameters, fl: np.ndarray) -> None:
        """Log fluctuations immediately when received."""
        if not self.tb_logger:
            return

        try:
            # Use the length of fluctuations history as step
            step = len(self.order_parameters_history[parameters].fluctuations) if parameters in self.order_parameters_history else None

            # Log individual fluctuation values for this temperature
            for field in self.tb_logger.order_parameter_fields:
                if len(fl) > 0:
                    value = float(fl[-1][field])  # Get the most recent value
                    self.tb_logger.log_temperature_scalar_auto('fluctuations', field, parameters, value, step)

        except Exception as e:
            logger.error(f"Error logging fluctuations to TensorBoard: {e}")

    def _log_state_to_tensorboard(self, state: LatticeState) -> None:
        """Log state information (energy, acceptance rates) immediately when received."""
        if not self.tb_logger:
            return

        try:
            # Log lattice energy
            energy = state.lattice_averages['energy']
            self.tb_logger.log_temperature_scalar_auto('lattice_averages', 'energy', state.parameters, float(energy), state.iterations)

            # Log acceptance rates
            if state.iterations > 0:
                assert state.lattice.particles is not None, "Lattice particles should not be None when logging state"

                # accepted_x and accepted_p are counts for the most recent iteration
                # Each iteration attempts to update all particles once
                particles_size = state.lattice.particles.size
                orientation_rate = state.accepted_x / particles_size
                parity_rate = state.accepted_p / particles_size

                self.tb_logger.log_temperature_scalar_auto('acceptance_rates', 'orientation', state.parameters, orientation_rate * 100, state.iterations)
                self.tb_logger.log_temperature_scalar_auto('acceptance_rates', 'parity', state.parameters, parity_rate * 100, state.iterations)

        except Exception as e:
            logger.error(f"Error logging state to TensorBoard: {e}")

    def _log_current_temperatures_after_exchange(self, pt_ready: List[Tuple[int, ParallelTemperingParameters]], step: Optional[int] = None) -> None:
        """Log current temperature for each simulation process after parallel tempering exchange."""
        if not self.tb_logger:
            return

        try:
            # After exchange, each process gets the temperature from the parameters that were sent to it
            # The pt_ready list contains (simulation_index, ParallelTemperingParameters) for each process
            for sim_index, pt_param in pt_ready:
                current_temp = float(pt_param.parameters.temperature)
                self.tb_logger.log_simulation_temperature_after_exchange_auto(sim_index, current_temp, step)
        except Exception as e:
            logger.error(f"Error logging temperatures after exchange to TensorBoard: {e}")

    def log_temperature_plots_to_tensorboard(self, recent_points: int = 1000) -> None:
        """
        Log temperature-based plots for energy, order parameters, and fluctuations.

        This method creates matplotlib plots showing how energy, order parameters, and
        fluctuations vary with temperature across all simulation processes. The plots
        are logged as images to TensorBoard for visualization.

        Parameters
        ----------
        recent_points : int, optional
            Number of recent data points to average for each temperature (default: 100)

        Examples
        --------
        # Log temperature plots during simulation
        runner.log_temperature_plots_to_tensorboard(recent_points=50)

        # Log with more data points for smoother averaging
        runner.log_temperature_plots_to_tensorboard(recent_points=200)
        """
        if not self.tb_logger:
            return

        try:
            import time
            current_step = int(time.time() - (self._start_time or time.time()))

            # Log all temperature-based plots
            self.tb_logger.log_all_temperature_plots(
                self.order_parameters_history, current_step, recent_points
            )

            logger.debug(f"Temperature-based plots logged at step {current_step}")

        except Exception as e:
            logger.error(f"Error logging temperature plots to TensorBoard: {e}")
