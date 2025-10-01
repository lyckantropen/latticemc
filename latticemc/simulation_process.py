"""Individual simulation process for parallel lattice Monte Carlo simulations."""

import logging
import multiprocessing as mp
from collections import namedtuple
from ctypes import c_double
from enum import Enum
from typing import List, Optional, cast

import numpy as np

from .definitions import LatticeState
from .simulation import Simulation
from .updaters import CallbackUpdater, Updater

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


class SimulationProcess(Simulation, mp.Process):
    """
    Process representing one simulation over a lattice of particles.

    Inherits from both mp.Process for multiprocessing and Simulation for
    the core simulation logic. Adds queue-based communication for parallel execution.
    """

    def __init__(self,
                 index: int,
                 queue: mp.Queue,
                 initial_state: LatticeState,
                 *args,
                 report_order_parameters_every: int = 1000,
                 report_fluctuations_every: int = 1000,
                 report_state_every: int = 1000,
                 parallel_tempering_interval: Optional[int] = None,
                 exchange_barrier=None,
                 parallel_tempering_timeout: int = 240,
                 auto_recover: bool = False,
                 **kwargs) -> None:
        # Disable progress bar for multiprocessing
        kwargs['progress_bar'] = None

        # Initialize parent classes - order matters for multiple inheritance
        Simulation.__init__(self, initial_state, *args, **kwargs)
        mp.Process.__init__(self)

        self.index = index
        self.queue = queue
        self.report_order_parameters_every = report_order_parameters_every
        self.report_fluctuations_every = report_fluctuations_every
        self.report_state_every = report_state_every
        self.parallel_tempering_interval = parallel_tempering_interval
        self.exchange_barrier = exchange_barrier
        self.parallel_tempering_timeout = parallel_tempering_timeout
        self.auto_recover = auto_recover

        self.running = mp.Value('i', 1)
        self.temperature = mp.Value(c_double, float(self.state.parameters.temperature))

        if self.auto_recover:
            self.recover()

    def _create_additional_updaters(self) -> List[Updater]:
        """Create additional updaters for queue communication and parallel tempering."""
        # Get any updaters from parent class
        super_updaters = super()._create_additional_updaters()

        updaters: List[Updater] = [*super_updaters]

        # Add parallel tempering updater if enabled
        if self.parallel_tempering_interval is not None and self.exchange_barrier is not None:
            parallel_tempering_updater = CallbackUpdater(
                callback=lambda _: self._parallel_tempering(),
                how_often=self.parallel_tempering_interval,
                since_when=self.parallel_tempering_interval
            )
            updaters.append(parallel_tempering_updater)

        # Queue communication updaters
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

        updaters += [
            order_parameters_broadcaster,
            fluctuations_broadcaster,
            state_broadcaster,
            ping_updater
        ]

        return updaters

    def _setup_process_logging(self) -> None:
        """Set up logging for this simulation process."""
        import logging

        # Ensure working_folder is not None (method should only be called when it's not None)
        assert self.working_folder is not None, "working_folder must be set before calling _setup_process_logging"

        # Create process-specific log file
        log_file = self.working_folder / "logs" / "simulation.log"

        # Set up file handler with UTF-8 encoding to handle Greek letters
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Create formatter with process and temperature identification
        temp = self.state.parameters.temperature
        lam = self.state.parameters.lam
        tau = self.state.parameters.tau
        process_id = f"P{self.index:03d}_T{temp}_λ{lam}_τ{tau}"

        formatter = logging.Formatter(
            f'%(asctime)s - {process_id} - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # Get root logger and add our handler
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)

        logging.info(f"Process {self.index} logging set up (T={temp}, λ={lam}, τ={tau})")

    def run(self) -> None:
        """Execute the simulation process by calling the Simulation.run() method."""
        # Set up process-specific logging if working folder is available
        if self.working_folder is not None:
            self._setup_process_logging()

        # Call the Simulation.run() method, not mp.Process.run()
        Simulation.run(self)

    def _handle_simulation_error(self, error: Exception) -> None:
        """Handle simulation errors by notifying the queue and calling parent handler."""
        self.queue.put((MessageType.Error, self.index, (self.state.parameters, error)))
        super()._handle_simulation_error(error)

    def _simulation_finished(self) -> None:
        """Handle simulation completion by marking as finished and notifying queue."""
        # Mark as finished - this will prevent future barrier waits
        with self.running.get_lock():
            self.running.value = 0

        # Call parent method to perform final save and cleanup
        super()._simulation_finished()

        self.queue.put((MessageType.State, self.index, self.state))
        self.queue.put((MessageType.Finished, self.index, self.state.parameters))

    def _broadcast_order_parameters(self) -> None:
        """Publish recent order parameters from history to the governing thread."""
        params = self.local_history.order_parameters
        self.queue.put((MessageType.OrderParameters, self.index,
                        (self.state.parameters, params)))
        self.local_history.order_parameters_list.clear()  # Clear after broadcasting

    def _broadcast_fluctuations(self) -> None:
        """Publish recent fluctuation values from history to the governing thread."""
        fluctuations = self.local_history.fluctuations
        self.queue.put((MessageType.Fluctuations, self.index,
                        (self.state.parameters, fluctuations)))
        self.local_history.fluctuations_list.clear()  # Clear after broadcasting

    def _broadcast_state(self):
        """Publish the current Lattice State to the governing thread."""
        self.queue.put((MessageType.State, self.index, self.state))

    def _send_ping(self) -> None:
        """Send a ping message to indicate the process is alive and healthy."""
        self.queue.put((MessageType.Ping, self.index, self.current_step))

    def _parallel_tempering(self) -> None:
        """Announce readiness for parallel tempering and perform exchange if selected."""
        # Check if we're still supposed to be running - if not, skip parallel tempering
        if not self.running.value:
            logger.debug(f'SimulationProcess[{self.index}, {self.state.parameters}]: Skipping parallel tempering - simulation ending')
            return

        # Wait at barrier to synchronize all replicas before attempting exchange
        if self.exchange_barrier is not None:
            logger.debug(f'SimulationProcess[{self.index}, {self.state.parameters}]: Waiting at exchange barrier')
            try:
                self.exchange_barrier.wait(timeout=self.parallel_tempering_timeout)  # Add timeout to prevent indefinite hanging
            except Exception as e:
                # Handle barrier broken, timeout, or other barrier-related exceptions
                logger.debug(f'SimulationProcess[{self.index}, {self.state.parameters}]: Barrier exception ({e}), skipping parallel tempering')
                return

        # energy needs to be scaled by number of particles (total system energy not per-particle)
        # the energy stored in order parameters is a lattice average
        if len(self.local_history.order_parameters_list) == 0:
            logger.error(f'SimulationProcess[{self.index}, {self.state.parameters}]: No order parameters available for parallel tempering, skipping exchange')
            return

        energy = self.local_history.order_parameters_list[-1]['energy'] * cast(np.ndarray, self.state.lattice.particles).size
        our, theirs = mp.Pipe()
        self.queue.put((MessageType.ParallelTemperingSignUp, self.index, ParallelTemperingParameters(
            parameters=self.state.parameters, energy=energy, pipe=theirs)))

        # wait for decision in governing thread
        if not our.poll(self.parallel_tempering_timeout):
            logger.warning(f'SimulationProcess[{self.index}, {self.state.parameters}]: No parallel tempering data to exchange')
            return
        parameters = our.recv()

        logger.debug(f'SimulationProcess[{self.index}, {self.state.parameters}]: Received parameters for exchange: {parameters}')

        if parameters != self.state.parameters:
            logger.debug(f'SimulationProcess[{self.index}, {self.state.parameters}]: Performing parallel tempering exchange to {parameters}')
            # broadcast what we can
            self._broadcast_order_parameters()
            self._broadcast_fluctuations()
            self._broadcast_state()

            # parameter change
            with self.temperature.get_lock():
                self.state.parameters = parameters
                self.temperature.value = float(parameters.temperature)

            # clear history to avoid mixing data from different temperatures
            logger.debug(f'SimulationProcess[{self.index}, {self.state.parameters}]:'
                         f' Clearing {len(self.local_history.order_parameters_list)} samples after parallel tempering exchange')
            self.local_history.clear()

    def recover(self) -> bool:
        """
        Override recover to broadcast recovered data to the governing process.

        After successful recovery, immediately broadcasts the recovered order parameters
        and fluctuations to the SimulationRunner so it can populate its history dictionary
        with the correct parameter keys.
        """
        # Call parent's recover method
        success = super().recover()

        # TODO: this is wrong. the order parameters history contains data for different temperatures.

        if success and self.working_folder is not None:
            logger.info(f"SimulationProcess[{self.index}]: Broadcasting recovered data to runner")

            # Broadcast recovered order parameters if available
            if len(self.local_history.order_parameters_list) > 0:
                try:
                    # Send all recovered order parameters
                    op_data = self.local_history.order_parameters  # All recovered
                    self.queue.put((MessageType.OrderParameters, self.index, (self.state.parameters, op_data)))
                    logger.debug(f"SimulationProcess[{self.index}]: Sent recovered order parameters")
                except Exception as e:
                    logger.warning(f"SimulationProcess[{self.index}]: Failed to broadcast recovered order parameters: {e}")

            # Broadcast recovered fluctuations if available
            if len(self.local_history.fluctuations_list) > 0:
                try:
                    # Send all recovered fluctuations
                    fl_data = self.local_history.fluctuations  # All recovered
                    self.queue.put((MessageType.Fluctuations, self.index, (self.state.parameters, fl_data)))
                    logger.debug(f"SimulationProcess[{self.index}]: Sent recovered fluctuations")
                except Exception as e:
                    logger.warning(f"SimulationProcess[{self.index}]: Failed to broadcast recovered fluctuations: {e}")

            # Also broadcast the current state so runner knows the actual parameters
            try:
                self.queue.put((MessageType.State, self.index, self.state))
                logger.debug(f"SimulationProcess[{self.index}]: Sent recovered state")
            except Exception as e:
                logger.warning(f"SimulationProcess[{self.index}]: Failed to broadcast recovered state: {e}")

            # Send ping
            try:
                self._send_ping()
                logger.debug(f"SimulationProcess[{self.index}]: Sent ping after recovery")
            except Exception as e:
                logger.warning(f"SimulationProcess[{self.index}]: Failed to send ping after recovery: {e}")

        return success
