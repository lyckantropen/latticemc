"""Base simulation class for lattice Monte Carlo simulations."""

import logging
from typing import List, Optional, Any

from .definitions import LatticeState, OrderParametersHistory
from .failsafe import failsafe_save_simulation
from .updaters import FluctuationsCalculator, OrderParametersCalculator, Updater

logger = logging.getLogger(__name__)


class Simulation:
    """
    Base class for lattice Monte Carlo simulations.

    Handles the core simulation loop, order parameter tracking, and standard updaters.
    Does not include multiprocessing or communication-specific functionality.

    Args
    ----
        initial_state: The initial lattice state for the simulation
        cycles: Number of Monte Carlo steps to perform
        fluctuations_window: Window size for fluctuation calculations
        per_state_updaters: Additional updaters to run each step
        progress_bar: Optional tqdm progress bar object. If None, creates automatic
                     console progress bar. Use tqdm.notebook.tqdm for Jupyter notebooks.
    """

    def __init__(self,
                 initial_state: LatticeState,
                 cycles: int,
                 fluctuations_window: int = 1000,
                 per_state_updaters: Optional[List[Updater]] = None,
                 progress_bar: Optional[Any] = None) -> None:
        self.state = initial_state
        self.cycles = cycles
        self.fluctuations_window = fluctuations_window
        self.per_state_updaters = per_state_updaters or []
        self.progress_bar = progress_bar

        self.local_history = OrderParametersHistory()
        self.current_step = 0

        # Track how many steps are relevant for current configuration
        # (important for parallel tempering when parameters change)
        self._relevant_history_length = 0

    def run(self) -> None:
        """Execute the simulation."""
        self.current_step = 0
        logger.info(f"Starting simulation with {self.cycles} cycles")
        logger.info(f"Parameters: T={self.state.parameters.temperature}, "
                    f"λ={self.state.parameters.lam}, τ={self.state.parameters.tau}")
        logger.debug(f"Fluctuations window: {self.fluctuations_window}")
        logger.debug(f"Additional updaters: {len(self.per_state_updaters)}")

        # Set up core updaters
        updaters = self._create_updaters()
        logger.debug(f"Total updaters created: {len(updaters)}")

        # MAIN SIMULATION LOOP
        try:
            from . import simulation_numba
            logger.debug("Starting main simulation loop")

            # Create progress bar if not provided
            progress_iterator = self._create_progress_iterator()

            for step in progress_iterator:
                simulation_numba.do_lattice_state_update(self.state)
                self._relevant_history_length += 1
                self.current_step = step + 1

                for updater in updaters:
                    updater.perform(self.state)

                # Update progress bar if available
                if self.progress_bar is not None:
                    # Update description with current energy if available
                    if len(self.local_history.order_parameters) > 0:
                        current_energy = self.local_history.order_parameters['energy'][-1]
                        self.progress_bar.set_description(f"Energy: {current_energy:.6f}")

                # Log progress periodically (fallback when no progress bar)
                elif (step + 1) % 1000 == 0:
                    logger.debug(f"Completed {step + 1}/{self.cycles} steps")
                    if len(self.local_history.order_parameters) > 0:
                        current_energy = self.local_history.order_parameters['energy'][-1]
                        logger.debug(f"Current energy: {current_energy:.6f}")

        except Exception as e:
            logger.error(f"Simulation failed at step {self.current_step}: {e}")
            self._handle_simulation_error(e)

        self._simulation_finished()

    def _create_progress_iterator(self):
        """Create a progress iterator using tqdm if available, otherwise plain range."""
        if self.progress_bar is not None:
            # Use the provided progress bar (could be tqdm, tqdm.notebook, etc.)
            return self.progress_bar
        else:
            # Try to import and use tqdm for console progress
            try:
                from tqdm import tqdm
                return tqdm(range(self.cycles), desc="Simulation Progress")
            except ImportError:
                # Fallback to plain range if tqdm not available
                logger.debug("tqdm not available, using plain progress logging")
                return range(self.cycles)

    def _create_updaters(self) -> List[Updater]:
        """Create the list of updaters for the simulation."""
        logger.debug("Creating simulation updaters")

        # Core data collection updaters
        order_parameters_calculator = OrderParametersCalculator(
            self.local_history, how_often=1, since_when=0
        )
        logger.debug("Added OrderParametersCalculator")

        fluctuations_calculator = FluctuationsCalculator(
            self.local_history,
            window=self.fluctuations_window,
            how_often=1,
            since_when=self.fluctuations_window
        )
        logger.debug(f"Added FluctuationsCalculator with window={self.fluctuations_window}")

        # Combine with user-provided updaters
        updaters = [
            order_parameters_calculator,
            fluctuations_calculator,
            *self.per_state_updaters,
        ]

        if self.per_state_updaters:
            logger.debug(f"Added {len(self.per_state_updaters)} per-state updaters")

        # Add any additional updaters from derived classes
        additional_updaters = self._create_additional_updaters()
        if additional_updaters:
            updaters.extend(additional_updaters)
            logger.debug(f"Added {len(additional_updaters)} additional updaters from derived class")

        logger.debug(f"Created {len(updaters)} total updaters")
        return updaters

    def _create_additional_updaters(self) -> List[Updater]:
        """
        Create additional updaters for derived classes.

        Returns
        -------
        List[Updater]
            List of additional updaters to include in the simulation loop.
        """
        return []

    def _handle_simulation_error(self, error: Exception) -> None:
        """
        Handle errors that occur during simulation.

        Args
        ----
            error: The exception that occurred during simulation.
        """
        logger.error(f"Simulation error at cycle {self._relevant_history_length}: {error}")
        logger.debug("Saving simulation state for debugging")
        failsafe_save_simulation(error, self.state, self.local_history)
        logger.info("Simulation state saved for debugging")

    def _simulation_finished(self) -> None:
        """
        Do stuff when simulation completes successfully.

        Derived classes can override this to perform cleanup or final actions.
        """
        logger.info(f"Simulation completed successfully after {self.cycles} cycles")
        logger.debug(f"Final relevant history length: {self._relevant_history_length}")
        if len(self.local_history.order_parameters) > 0:
            final_energy = self.local_history.order_parameters['energy'][-1]
            logger.debug(f"Final energy: {final_energy:.6f}")
        logger.debug(f"Total collected data points: {len(self.local_history.order_parameters)}")

    def reset_relevant_history(self) -> None:
        """
        Reset the relevant history length counter.

        This is typically called when parameters change (e.g., in parallel tempering)
        to indicate that only recent history is relevant to the current configuration.
        """
        old_length = self._relevant_history_length
        self._relevant_history_length = 0
        logger.debug(f"Reset relevant history length from {old_length} to 0")

    def get_recent_order_parameters(self, max_count: Optional[int] = None):
        """Get recent order parameters, limited by relevant history length."""
        if max_count is None:
            max_count = self._relevant_history_length

        count = min(self._relevant_history_length, max_count)
        if count > 0:
            return self.local_history.order_parameters[-count:]
        return self.local_history.order_parameters

    def get_recent_fluctuations(self, max_count: Optional[int] = None):
        """Get recent fluctuation values, limited by relevant history length."""
        if max_count is None:
            max_count = self._relevant_history_length

        count = min(self._relevant_history_length, max_count)
        if count > 0:
            return self.local_history.fluctuations[-count:]
        return self.local_history.fluctuations
