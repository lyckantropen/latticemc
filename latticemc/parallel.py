"""Parallel processing and threading support for lattice simulations."""

import json
import logging
import multiprocessing as mp
import pathlib
import queue
import threading
import time
from collections import defaultdict
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from .definitions import DefiningParameters, LatticeState, OrderParametersHistory
from .simulation_process import MessageType, ParallelTemperingParameters, SimulationProcess
from .tensorboard_logger import TensorBoardLogger

logger = logging.getLogger(__name__)


class ProgressBarMode(Enum):
    """Progress bar display modes for parallel simulations."""

    NONE = "none"
    CONSOLE = "console"
    NOTEBOOK = "notebook"


class SimulationRunner(threading.Thread):
    """
    Thread that manages multiple simulation processes with parallel tempering support.

    SimulationRunner coordinates multiple SimulationProcess instances, handles inter-process
    communication, manages parallel tempering exchanges, and provides comprehensive data
    logging and visualization through TensorBoard integration.

    Parameters
    ----------
    initial_states : List[LatticeState]
        List of initial lattice states for each simulation process. Each state should
        have different parameters (typically temperature) for parallel tempering.
    order_parameters_history : Dict[DefiningParameters, OrderParametersHistory]
        Dictionary to collect and store results from each parameter set. Organized
        by DefiningParameters for efficient analysis across temperature exchanges.
    cycles : int
        Number of Monte Carlo steps each process should perform.
    parallel_tempering_interval : Optional[int], default=None
        Interval (in steps) for parallel tempering exchanges. If None, parallel
        tempering is disabled and processes run independently.
    progress_bar_mode : ProgressBarMode, default=ProgressBarMode.CONSOLE
        Progress bar display mode: NONE (no bars), CONSOLE (terminal), or
        NOTEBOOK (Jupyter notebook compatible).
    working_folder : Optional[str], default=None
        Path to folder for saving simulation states, data, and logs. Creates
        organized structure with process folders, parameter folders, and plots.
        If None, no file saving is performed.
    save_interval : int, default=200
        How often to save simulation state and data (in steps). Controls both
        process-level and parameter-level saving frequency.
    auto_recover : bool, default=False
        Whether to attempt recovery from previously saved state when working_folder
        is specified and recovery markers are found.
    plot_save_interval_sec : float, default=60.0
        How often to generate and save temperature plots (in seconds of real time).
        Plots are saved to both TensorBoard and working folder if available.
    plot_recent_points : int, default=1000
        Number of recent data points to use for temperature plot generation.
        Larger values provide more statistical accuracy but slower plotting.
    **kwargs
        Additional keyword arguments passed to SimulationProcess instances.
        See SimulationProcess documentation for supported options.

    Folder Structure Created
    ------------------------
    When working_folder is specified, creates the following structure:

    working_folder/
    ├── tensorboard/                    # TensorBoard logs for real-time visualization
    ├── plots/                          # Temperature-dependent plots
    │   ├── energy_vs_temperature.png
    │   └── order_parameter_*_vs_temperature.png
    ├── logs/                           # Simulation runner logs
    ├── process_000/, process_001/, ... # Individual process data and states
    │   ├── data/
    │   │   ├── order_parameters.npz
    │   │   └── fluctuations.npz
    │   ├── states/
    │   │   └── latest_state.npz
    │   ├── logs/
    │   │   └── simulation.log
    │   └── summary.json
    └── parameters/                    # Data organized by actual parameter values
        ├── T0.30_lam0.35_tau1.00/
        ├── T0.35_lam0.35_tau1.00/
        └── T1.65_lam0.35_tau1.00/
            ├── data/
            │   ├── order_parameters.npz
            │   └── fluctuations.npz
            ├── states/
            │   └── latest_state.npz
            └── summary.json

    Notes
    -----
    - For parallel tempering, provide states with different temperatures
    - Process communication uses multiprocessing.Queue for thread safety
    - TensorBoard logging is automatically enabled when working_folder is provided
    - Temperature plots are generated periodically and saved permanently
    - All data is organized both by process ID and by actual parameter values
    - Recovery markers prevent data corruption during simulation restarts
    """

    def __init__(self,
                 initial_states: List[LatticeState],
                 order_parameters_history: Dict[DefiningParameters, OrderParametersHistory],
                 cycles: int,
                 *args,
                 parallel_tempering_interval: Optional[int] = None,
                 progress_bar_mode: ProgressBarMode = ProgressBarMode.CONSOLE,
                 working_folder: Optional[str] = None,
                 save_interval: int = 200,
                 auto_recover: bool = False,
                 plot_save_interval_sec: float = 60.0,
                 plot_recent_points: int = 1000,
                 **kwargs) -> None:
        threading.Thread.__init__(self)

        self.cycles = cycles
        self.states = initial_states
        self.order_parameters_history = order_parameters_history
        self.args = args
        self.simulations: List[SimulationProcess] = []
        self.plot_save_interval_sec = plot_save_interval_sec
        self.plot_recent_points = plot_recent_points
        self.kwargs = kwargs
        self.progress_bar_mode = progress_bar_mode
        self.working_folder = working_folder
        self.save_interval = save_interval
        self.auto_recover = auto_recover

        # set to False when all processes have started running
        self._starting = True

        # Track process errors and failures
        self._process_errors: List[Exception] = []
        self._crashed_processes: List[int] = []

        # Track thread-level exceptions (for communicating back to main thread)
        self._thread_exception: Optional[Exception] = None

        # Track process health via ping messages
        self._last_ping_time: Dict[int, float] = {}
        self._ping_timeout = 120.0  # seconds without ping before considering process dead

        # Caching for alive() checks to reduce process polling overhead
        self._last_alive_check_time = 0.0
        self._last_alive_result = True
        self._alive_check_interval = 0.01  # Cache alive() result for 10ms

        # Progress bar support
        self._progress_meters: Dict[int, int] = defaultdict(lambda: 0)  # counting number of iterations per process reported by Ping
        self._progress_bars: Optional[Dict[str, Any]] = None  # for tqdm objects

        # Working folder support

        # for parallel tempering
        self._temperatures: List[Decimal] = [state.parameters.temperature for state in self.states]
        self._exchange_counter = 0  # Global counter for replica exchange rounds

        # barrier for synchronization during replica exchange
        # Create barrier if parallel tempering is enabled and there are multiple replicas
        self._parallel_tempering_enabled = parallel_tempering_interval is not None
        self.kwargs['parallel_tempering_interval'] = parallel_tempering_interval
        self._exchange_barrier = None
        if self._parallel_tempering_enabled and len(self.states) > 1:
            self._exchange_barrier = mp.Barrier(len(self.states))

        # TensorBoard logging setup - always enabled
        self._tb_logger: Optional[TensorBoardLogger]
        try:
            # Use working_folder/tensorboard if working_folder is specified, otherwise use default
            tb_log_dir = None
            if self.working_folder is not None:
                tb_log_dir = f"{self.working_folder}/tensorboard"

            self._tb_logger = TensorBoardLogger(log_dir=tb_log_dir)
            logger.info(f"TensorBoard logging enabled, writing to {self._tb_logger.log_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize TensorBoard logging: {e}")
            self._tb_logger = None

        # Start time for logging
        self._start_time: Optional[float] = None

        # Data saving tracking - count received messages for each parameter set
        self._order_parameters_received_count: Dict[DefiningParameters, int] = defaultdict(int)
        self._fluctuations_received_count: Dict[DefiningParameters, int] = defaultdict(int)
        self._states_received_count: Dict[DefiningParameters, int] = defaultdict(int)

    def _create_progress_bars(self) -> None:
        """Create progress bars (tqdm objects) based on the specified mode."""
        if self.progress_bar_mode == ProgressBarMode.NONE:
            return

        try:
            if self.progress_bar_mode == ProgressBarMode.NOTEBOOK:
                from tqdm.notebook import tqdm
            else:  # CONSOLE mode
                from tqdm import tqdm

            self._progress_bars = {}
            total_cycles = self.cycles * len(self.states)

            # Create main progress bar
            main_kwargs = {
                "total": total_cycles,
                "desc": "Overall Progress",
                "unit": "steps"
            }
            if self.progress_bar_mode == ProgressBarMode.CONSOLE:
                main_kwargs["position"] = 0

            self._progress_bars["main"] = tqdm(**main_kwargs)

            # Create per-simulation progress bars
            for i in range(len(self.states)):
                temp = float(self.states[i].parameters.temperature)
                sim_kwargs = {
                    "total": self.cycles,
                    "desc": f"Sim {i} (T={temp:.3f})",
                    "unit": "steps",
                    "leave": False
                }
                if self.progress_bar_mode == ProgressBarMode.CONSOLE:
                    sim_kwargs["position"] = i + 1

                self._progress_bars[f"sim_{i}"] = tqdm(**sim_kwargs)

        except ImportError as e:
            logger.warning(f"tqdm not available ({e}), progress bars disabled")
            self.progress_bar_mode = ProgressBarMode.NONE

    def _update_progress_bars(self) -> None:
        """Update all progress bars (tqdm objects) based on current _progress_meters."""
        if self._progress_bars is None:
            return

        # Update main progress bar
        if "main" in self._progress_bars:
            total_progress = sum(self._progress_meters.values())
            self._progress_bars["main"].n = total_progress
            self._progress_bars["main"].refresh()

        # Update individual simulation progress bars
        for sim_index, progress in self._progress_meters.items():
            sim_key = f"sim_{sim_index}"
            if sim_key in self._progress_bars:
                self._progress_bars[sim_key].n = progress
                self._progress_bars[sim_key].refresh()

    def _cleanup_progress_bars(self) -> None:
        """Clean up created progress bars."""
        if self._progress_bars is None:
            return

        for pbar in self._progress_bars.values():
            try:
                pbar.close()
            except Exception as e:
                logger.debug(f"Error closing progress bar: {e}")

        self._progress_bars = None

    def run(self) -> None:
        """Execute all simulation processes."""
        q: mp.Queue = mp.Queue()

        # Log all temperature assignments to TensorBoard
        if self._tb_logger:
            self._tb_logger.log_simulation_temperature_assignments(self.states)

        self._start_time = time.time()
        last_logged_time = self._start_time

        # Create progress bars if needed
        self._create_progress_bars()

        # create and start all simulations
        self.simulations = []
        for i, state in enumerate(self.states):
            # Prepare kwargs with barrier and working folder
            sim_kwargs = dict(self.kwargs)
            sim_kwargs['exchange_barrier'] = self._exchange_barrier
            sim_kwargs['cycles'] = self.cycles

            # Add working folder support - each process gets its own subfolder
            if self.working_folder is not None:
                process_folder = f"{self.working_folder}/process_{i:03d}"
                sim_kwargs['working_folder'] = process_folder
                sim_kwargs['save_interval'] = self.save_interval
                sim_kwargs['auto_recover'] = self.auto_recover

            sim = SimulationProcess(i, q, state, *self.args, **sim_kwargs)
            sim.start()
            self.simulations.append(sim)

            # Log the temperature assignment for each simulation index
            logger.info(f"SimulationProcess[{i}]: Running at temperature T={float(state.parameters.temperature):.4f}")

        self._starting = False

        # main message processing loop
        pt_ready: List[Tuple[int, ParallelTemperingParameters]] = []
        last_crash_check_time = time.time()
        crash_check_interval = 1.0  # Check for crashed processes every 1 second instead of every loop

        while self.alive():
            # Process messages with timeout to avoid busy-waiting
            try:
                # Use blocking get with timeout instead of busy-waiting on empty()
                message_type, index, msg = q.get(timeout=0.01)  # Block for up to 10ms

                logger.debug(f'SimulationRunner: Received {message_type}, index={index}')

                if message_type == MessageType.OrderParameters:
                    parameters, op = msg
                    self.order_parameters_history[parameters].append_order_parameters(op)
                    # Log order parameters immediately when received
                    self._log_order_parameters_to_tensorboard(parameters, op)

                    # Save order parameters periodically
                    self._order_parameters_received_count[parameters] += 1
                    if self._order_parameters_received_count[parameters]:
                        self._save_order_parameters(parameters)
                        self._save_parameter_summary(parameters)
                if message_type == MessageType.Fluctuations:
                    parameters, fl = msg
                    self.order_parameters_history[parameters].append_fluctuations(fl)
                    # Log fluctuations immediately when received
                    self._log_fluctuations_to_tensorboard(parameters, fl)

                    # Save fluctuations periodically
                    self._fluctuations_received_count[parameters] += 1
                    if self._fluctuations_received_count[parameters]:
                        self._save_fluctuations(parameters)
                        self._save_parameter_summary(parameters)
                if message_type == MessageType.State:
                    # update the state
                    msg = cast(LatticeState, msg)
                    state = [state for state in self.states if state.parameters == msg.parameters][0]
                    state.update_from(msg)
                    # Log energy and acceptance rates immediately when state is received
                    self._log_state_to_tensorboard(state)

                    # Save state periodically
                    parameters = msg.parameters
                    self._states_received_count[parameters] += 1
                    if self._states_received_count[parameters]:
                        self._save_state(parameters)
                        self._save_parameter_summary(parameters)
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
                    self._progress_meters[index] = iterations
                    logger.debug(f'SimulationProcess[{index}]: Ping received at iteration {iterations}')

                    # Update progress bars
                    self._update_progress_bars()

            except queue.Empty:
                # No messages available - this is normal, continue to next iteration
                pass

            # Check for crashed processes periodically, not every loop iteration
            current_time = time.time()
            if current_time - last_crash_check_time >= crash_check_interval:
                try:
                    self._check_for_crashed_processes()
                    last_crash_check_time = current_time
                except RuntimeError as e:
                    logger.error(f'SimulationRunner: Stopping due to error: {e}')
                    self._thread_exception = e
                    self.stop()
                    break  # Exit the main loop instead of raising

            # Perform parallel tempering if enabled and there are replicas ready
            if self._parallel_tempering_enabled:
                self._do_parallel_tempering(pt_ready)

            # Log simulation progress every self.plot_save_interval_sec seconds
            if self._tb_logger:
                current_time = time.time()
                if current_time - last_logged_time >= self.plot_save_interval_sec:
                    self._save_temperature_plots(recent_points=self.plot_recent_points)
                    last_logged_time = current_time

        # Close TensorBoard logger
        if self._tb_logger:
            self._tb_logger.close()

        # Ensure all processes have finished
        [sim.join() for sim in self.simulations]  # type: ignore

        # Save final temperature plots
        self._save_temperature_plots(recent_points=self.cycles, fluctuations_from_history=True)

        # Save final data for all parameter sets
        self._save_final_data()

        # Clean up progress bars
        self._cleanup_progress_bars()

        # Save final JSON summary file
        self._save_final_summary()

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
        if self._tb_logger:
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

        # Clean up progress bars
        self._cleanup_progress_bars()

    def alive(self) -> bool:
        """Check if any simulation processes are still running with caching to reduce overhead."""
        current_time = time.time()

        # Use cached result if checked recently
        if current_time - self._last_alive_check_time < self._alive_check_interval:
            return self._last_alive_result

        # Update cache
        self._last_alive_check_time = current_time
        self._last_alive_result = self._starting or bool([sim for sim in self.simulations if sim.is_alive()])
        return self._last_alive_result

    def has_thread_exception(self) -> bool:
        """Check if the SimulationRunner thread encountered an exception."""
        return self._thread_exception is not None

    def get_thread_exception(self) -> Optional[Exception]:
        """Get the exception that occurred in the SimulationRunner thread, if any."""
        return self._thread_exception

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

        # Check if there was a thread-level exception
        if self.has_thread_exception():
            print(f"Simulation did not finish gracefully due to thread exception: {self._thread_exception}")
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
        if not self._tb_logger:
            return

        try:
            # Use the length of order parameters history as step
            step = len(self.order_parameters_history[parameters].order_parameters_list) if parameters in self.order_parameters_history else None

            # Log individual order parameter values for this temperature
            for field in self._tb_logger.order_parameter_fields:
                if len(op) > 0:
                    value = float(op[-1][field])  # Get the most recent value
                    self._tb_logger.log_temperature_scalar_auto('order_parameters', field, parameters, value, step)

        except Exception as e:
            logger.error(f"Error logging order parameters to TensorBoard: {e}")

    def _log_fluctuations_to_tensorboard(self, parameters: DefiningParameters, fl: np.ndarray) -> None:
        """Log fluctuations immediately when received."""
        if not self._tb_logger:
            return

        try:
            # Use the length of fluctuations history as step
            step = len(self.order_parameters_history[parameters].fluctuations_list) if parameters in self.order_parameters_history else None

            # Log individual fluctuation values for this temperature
            for field in self._tb_logger.order_parameter_fields:
                if len(fl) > 0:
                    value = float(fl[-1][field])  # Get the most recent value
                    self._tb_logger.log_temperature_scalar_auto('fluctuations', field, parameters, value, step)

        except Exception as e:
            logger.error(f"Error logging fluctuations to TensorBoard: {e}")

    def _log_state_to_tensorboard(self, state: LatticeState) -> None:
        """Log state information (energy, acceptance rates) immediately when received."""
        if not self._tb_logger:
            return

        try:
            # Log lattice energy
            energy = state.lattice_averages['energy']
            self._tb_logger.log_temperature_scalar_auto('lattice_averages', 'energy', state.parameters, float(energy), state.iterations)

            # Log acceptance rates
            if state.iterations > 0:
                assert state.lattice.particles is not None, "Lattice particles should not be None when logging state"

                # accepted_x and accepted_p are counts for the most recent iteration
                # Each iteration attempts to update all particles once
                particles_size = state.lattice.particles.size
                orientation_rate = state.accepted_x / particles_size
                parity_rate = state.accepted_p / particles_size

                self._tb_logger.log_temperature_scalar_auto('acceptance_rates', 'orientation', state.parameters, orientation_rate * 100, state.iterations)
                self._tb_logger.log_temperature_scalar_auto('acceptance_rates', 'parity', state.parameters, parity_rate * 100, state.iterations)

        except Exception as e:
            logger.error(f"Error logging state to TensorBoard: {e}")

    def _log_current_temperatures_after_exchange(self, pt_ready: List[Tuple[int, ParallelTemperingParameters]], step: Optional[int] = None) -> None:
        """Log current temperature for each simulation process after parallel tempering exchange."""
        if not self._tb_logger:
            return

        try:
            # After exchange, each process gets the temperature from the parameters that were sent to it
            # The pt_ready list contains (simulation_index, ParallelTemperingParameters) for each process
            for sim_index, pt_param in pt_ready:
                current_temp = float(pt_param.parameters.temperature)
                self._tb_logger.log_simulation_temperature_after_exchange_auto(sim_index, current_temp, step)
        except Exception as e:
            logger.error(f"Error logging temperatures after exchange to TensorBoard: {e}")

    def _save_temperature_plots(self, recent_points: int = 1000, fluctuations_from_history: bool = False) -> None:
        """
        Save temperature-based plots for energy, order parameters, and fluctuations.

        This method creates plots showing how energy, order parameters, and
        fluctuations vary with temperature across all simulation processes. The plots
        are saved both to TensorBoard for visualization and to the working folder as
        PNG files for permanent storage. Uses the entire data history for analysis.

        Examples
        --------
        # Save temperature plots during simulation
        runner.save_temperature_plots()
        """
        try:
            import pathlib
            import time

            from .plot_utils import create_all_temperature_plots

            current_step = int(time.time() - (self._start_time or time.time()))

            # Create all temperature plots using plot_utils (uses entire history)
            all_plots = create_all_temperature_plots(
                self.order_parameters_history,
                recent_points=recent_points,
                fluctuations_from_history=fluctuations_from_history)

            # Log plots to TensorBoard if available
            if self._tb_logger:
                for plot_name, img in all_plots.items():
                    img_array = np.array(img)
                    # TensorBoard expects CHW format (Color, Height, Width)
                    if len(img_array.shape) == 3:
                        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
                    tag = f'{plot_name}_vs_temperature'
                    self._tb_logger.writer.add_image(tag, img_array, current_step, dataformats='CHW')

                logger.debug(f"Temperature-based plots logged to TensorBoard at step {current_step}")

            # Save plots to working folder if available
            if self.working_folder and all_plots:
                plots_dir = pathlib.Path(self.working_folder) / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)

                for plot_name, img in all_plots.items():
                    plot_path = plots_dir / f"{plot_name}_vs_temperature.png"
                    img.save(plot_path, format='PNG', dpi=(150, 150))

                logger.debug("Temperature-based plots saved to working folder")

        except Exception as e:
            logger.error(f"Error saving temperature plots: {e}")

    def _get_parameter_save_paths(self, parameters: DefiningParameters) -> Dict[str, pathlib.Path]:
        """Get save paths organized by DefiningParameters.

        Returns paths for saving order parameters, fluctuations, and states
        in folders named by the defining parameters.
        """
        if self.working_folder is None:
            return {}

        param_folder_name = parameters.get_folder_name()
        base_path = pathlib.Path(self.working_folder) / "parameters" / param_folder_name

        return {
            'order_parameters': base_path / "data" / "order_parameters.npz",
            'fluctuations': base_path / "data" / "fluctuations.npz",
            'state': base_path / "states" / "latest_state.npz",
            'summary': base_path / "summary.json"
        }

    def _save_order_parameters(self, parameters: DefiningParameters) -> None:
        """Save order parameters for a specific parameter set."""
        if self.working_folder is None:
            return

        try:
            paths = self._get_parameter_save_paths(parameters)
            if not paths:
                return

            # Ensure directory exists
            paths['order_parameters'].parent.mkdir(parents=True, exist_ok=True)

            # Get order parameters history for this parameter set
            history = self.order_parameters_history[parameters]
            if history and len(history.order_parameters_list) > 0:
                history.save_to_npz(order_parameters_path=str(paths['order_parameters']))
                logger.debug(f"Saved order parameters for {parameters} to {paths['order_parameters']}")

        except Exception as e:
            logger.error(f"Error saving order parameters for {parameters}: {e}")

    def _save_fluctuations(self, parameters: DefiningParameters, fluctuations_from_history: bool = False) -> None:
        """Save fluctuations for a specific parameter set."""
        if self.working_folder is None:
            return

        try:
            paths = self._get_parameter_save_paths(parameters)
            if not paths:
                return

            # Ensure directory exists
            paths['fluctuations'].parent.mkdir(parents=True, exist_ok=True)

            # Get fluctuations history for this parameter set
            history = self.order_parameters_history[parameters]
            if history and len(history.fluctuations_list) > 0:
                history.save_to_npz(fluctuations_path=str(paths['fluctuations']), fluctuations_from_history=fluctuations_from_history)
                logger.debug(f"Saved fluctuations for {parameters} to {paths['fluctuations']}")

        except Exception as e:
            logger.error(f"Error saving fluctuations for {parameters}: {e}")

    def _save_state(self, parameters: DefiningParameters) -> None:
        """Save the latest state for a specific parameter set."""
        if self.working_folder is None:
            return

        try:
            paths = self._get_parameter_save_paths(parameters)
            if not paths:
                return

            # Find the state with matching parameters
            matching_state = None
            for state in self.states:
                if state.parameters == parameters:
                    matching_state = state
                    break

            if matching_state is None:
                logger.warning(f"No state found for parameters {parameters}")
                return

            # Ensure directory exists
            paths['state'].parent.mkdir(parents=True, exist_ok=True)

            # Save lattice state using LatticeState method
            if matching_state.lattice.particles is not None:
                save_data = matching_state.to_npz_dict(include_lattice=False, include_parameters=False)
                # Add only particles from lattice (not full lattice data to avoid duplication)
                save_data['particles'] = matching_state.lattice.particles

                np.savez_compressed(paths['state'], **save_data)
                logger.debug(f"Saved state for {parameters} to {paths['state']}")

        except Exception as e:
            logger.error(f"Error saving state for {parameters}: {e}")

    def _save_parameter_summary(self, parameters: DefiningParameters) -> None:
        """Save a JSON summary for a specific parameter set similar to Simulation class."""
        if self.working_folder is None:
            return

        try:
            paths = self._get_parameter_save_paths(parameters)
            if not paths:
                return

            # Find the state with matching parameters
            matching_state = None
            for state in self.states:
                if state.parameters == parameters:
                    matching_state = state
                    break

            if matching_state is None:
                return

            # Ensure directory exists
            paths['summary'].parent.mkdir(parents=True, exist_ok=True)

            # Get history for this parameter set
            history = self.order_parameters_history.get(parameters)

            # Create summary data similar to Simulation class structure
            summary_data = {
                'current_step': int(matching_state.iterations),
                'total_cycles': self.cycles,
                'parameters': parameters.to_dict()
            }

            # Add order parameters history data using its to_dict method
            if history:
                history_data = history.to_dict()
                summary_data.update(history_data)
            else:
                # Default empty data when no history available
                summary_data.update({
                    'latest_order_parameters': {},
                    'latest_fluctuations': {},
                    'data_counts': {
                        'order_parameters': 0,
                        'fluctuations': 0
                    }
                })

            # Add lattice state data using its to_dict method
            state_data = matching_state.to_dict()
            summary_data.update(state_data)

            # Save JSON summary
            with open(paths['summary'], 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved summary for {parameters} to {paths['summary']}")

        except Exception as e:
            logger.error(f"Error saving summary for {parameters}: {e}")

    def _save_final_data(self) -> None:
        """Save final data for all parameter sets when simulation completes."""
        if self.working_folder is None:
            return

        logger.info("Saving final data for all parameter sets...")

        # Save data for all parameter sets that have data
        for parameters in self.order_parameters_history.keys():
            try:
                self._save_order_parameters(parameters)
                self._save_fluctuations(parameters, fluctuations_from_history=True)
                self._save_state(parameters)
                self._save_parameter_summary(parameters)
                logger.info(f"Saved final data for parameters {parameters.get_folder_name()}")
            except Exception as e:
                logger.error(f"Error saving final data for {parameters}: {e}")

        logger.info("Final data saving completed.")

    def _save_final_summary(self) -> None:
        """Save a final JSON summary file for the entire simulation run."""
        if self.working_folder is None:
            return

        try:
            summary_path = Path(self.working_folder) / "simulation_summary.json"

            summary_data = {
                'total_processes': len(self.simulations),
                'total_cycles': self.cycles,
                'parameters_list': [state.parameters.to_dict() for state in self.states],
                'process_status': self.get_process_status(),
                'finished_gracefully': self.finished_gracefully(),
                'running_time_seconds': time.time() - (self._start_time or time.time())
            }

            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved final simulation summary to {summary_path}")

        except Exception as e:
            logger.error(f"Error saving final simulation summary: {e}")
