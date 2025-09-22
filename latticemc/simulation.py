"""Base simulation class for lattice Monte Carlo simulations."""

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

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
        working_folder: Optional path to folder for saving simulation state and logs.
                       If None, no saving is performed.
        save_interval: How often to save simulation state and JSON summary (in steps). Default 200.
        auto_recover: Whether to attempt recovery from saved state. Default False.
    """

    def __init__(self,
                 initial_state: LatticeState,
                 cycles: int,
                 fluctuations_window: int = 1000,
                 per_state_updaters: Optional[List[Updater]] = None,
                 progress_bar: Optional[Any] = None,
                 working_folder: Optional[str] = None,
                 save_interval: int = 200,
                 auto_recover: bool = False) -> None:
        self.state = initial_state
        self.cycles = cycles
        self.fluctuations_window = fluctuations_window
        self.per_state_updaters = per_state_updaters or []
        self.progress_bar = progress_bar

        # Persistence settings
        self.working_folder = pathlib.Path(working_folder) if working_folder else None
        self.save_interval = save_interval
        self.auto_recover = auto_recover

        self.local_history = OrderParametersHistory()
        self.current_step = 0

        # Track how many steps are relevant for current configuration
        # (important for parallel tempering when parameters change)
        self._relevant_history_length = 0

        # Set up working folder if requested
        self._setup_working_folder()

    def run(self) -> None:
        """Execute the simulation."""
        # Attempt recovery if enabled
        recovered = self.attempt_recovery()
        if not recovered:
            self.current_step = 0

        # Create marker to indicate simulation is in progress
        self._create_recovery_marker()

        logger.info(f"Starting simulation with {self.cycles} cycles")
        if recovered:
            logger.info(f"Resuming from step {self.current_step}")
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

            # Create progress bar if not provided - adjust for recovery
            start_step = self.current_step
            remaining_cycles = self.cycles - start_step
            progress_iterator = self._create_progress_iterator(remaining_cycles)

            for step in progress_iterator:
                # Calculate absolute step number
                absolute_step = start_step + step

                simulation_numba.do_lattice_state_update(self.state)
                self._relevant_history_length += 1
                self.current_step = absolute_step + 1

                for updater in updaters:
                    updater.perform(self.state)

                # Update progress bar if available
                if self.progress_bar is not None:
                    # Update description with current energy if available
                    if len(self.local_history.order_parameters_list) > 0:
                        current_energy = self.local_history.order_parameters_list[-1]['energy']
                        self.progress_bar.set_description(f"Energy: {current_energy:.6f}")

                # Log progress periodically (fallback when no progress bar)
                elif (step + 1) % 1000 == 0:
                    logger.debug(f"Completed {step + 1}/{self.cycles} steps")
                    if len(self.local_history.order_parameters_list) > 0:
                        current_energy = self.local_history.order_parameters_list[-1]['energy']
                        logger.debug(f"Current energy: {current_energy:.6f}")

        except Exception as e:
            logger.error(f"Simulation failed at step {self.current_step}: {e}")
            self._handle_simulation_error(e)

        self._simulation_finished()

    def _create_progress_iterator(self, remaining_cycles: Optional[int] = None):
        """Create a progress iterator using tqdm if available, otherwise plain range."""
        if remaining_cycles is None:
            remaining_cycles = self.cycles

        if self.progress_bar is not None:
            # Use the provided progress bar (could be tqdm, tqdm.notebook, etc.)
            return self.progress_bar
        else:
            # Try to import and use tqdm for console progress
            try:
                from tqdm import tqdm
                desc = f"Simulation Progress ({self.cycles - remaining_cycles}/{self.cycles})"
                return tqdm(range(remaining_cycles), desc=desc)
            except ImportError:
                # Fallback to plain range if tqdm not available
                logger.debug("tqdm not available, using plain progress logging")
                return range(remaining_cycles)

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
        Create additional updaters for derived classes and persistence.

        Returns
        -------
        List[Updater]
            List of additional updaters to include in the simulation loop.
        """
        additional_updaters: List[Updater] = []

        # Add persistence updaters if working folder is specified
        if self.working_folder is not None:
            paths = self._get_save_paths()

            # Order parameters saver
            op_saver = OrderParametersSaver(
                self.local_history,
                paths['order_parameters'],
                how_often=self.save_interval,
                since_when=self.save_interval
            )
            additional_updaters.append(op_saver)

            # Fluctuations saver
            fluct_saver = FluctuationsSaver(
                self.local_history,
                paths['fluctuations'],
                how_often=self.save_interval,
                since_when=self.save_interval
            )
            additional_updaters.append(fluct_saver)

            # Lattice state saver
            lattice_saver = LatticeStateSaver(
                paths['lattice_state'],
                self,
                how_often=self.save_interval,
                since_when=self.save_interval
            )
            additional_updaters.append(lattice_saver)

            # Simulation state saver
            sim_state_saver = SimulationStateSaver(
                paths['simulation_state'],
                self,
                how_often=self.save_interval,
                since_when=self.save_interval
            )
            additional_updaters.append(sim_state_saver)

            # JSON summary saver
            json_saver = JSONSummarySaver(
                self.local_history,
                paths['json_summary'],
                self,
                how_often=self.save_interval,
                since_when=self.save_interval
            )
            additional_updaters.append(json_saver)

            logger.debug(f"Added {len(additional_updaters)} persistence updaters")

        return additional_updaters

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
        if len(self.local_history.order_parameters_list) > 0:
            final_energy = self.local_history.order_parameters_list[-1]['energy']
            logger.debug(f"Final energy: {final_energy:.6f}")
        logger.debug(f"Total collected data points: {len(self.local_history.order_parameters_list)}")

        # Final save and cleanup if working folder is specified
        if self.working_folder is not None:
            logger.info("Performing final save and cleanup")
            self.save_simulation_data()
            self.save_json_summary()
            self._remove_recovery_marker()
            logger.info("Final save completed")

    def reset_relevant_history(self) -> None:
        """
        Reset the relevant history length counter.

        This is typically called when parameters change (e.g., in parallel tempering)
        to indicate that only recent history is relevant to the current configuration.
        """
        old_length = self._relevant_history_length
        self._relevant_history_length = 0
        logger.debug(f"Reset relevant history length from {old_length} to 0")

    def _setup_working_folder(self) -> None:
        """Set up the working folder for saving simulation data."""
        if self.working_folder is None:
            logger.debug("No working folder specified, skipping persistence setup")
            return

        # Create working folder if it doesn't exist
        self.working_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Working folder set up at: {self.working_folder}")

        # Create subdirectories
        (self.working_folder / "data").mkdir(exist_ok=True)
        (self.working_folder / "states").mkdir(exist_ok=True)
        (self.working_folder / "logs").mkdir(exist_ok=True)
        logger.debug("Created subdirectories: data, states, logs")

    def _get_save_paths(self) -> Dict[str, pathlib.Path]:
        """Get standard file paths for saving simulation data."""
        if self.working_folder is None:
            return {}

        return {
            'order_parameters': self.working_folder / "data" / "order_parameters.npz",
            'fluctuations': self.working_folder / "data" / "fluctuations.npz",
            'lattice_state': self.working_folder / "states" / "lattice_state.npz",
            'simulation_state': self.working_folder / "states" / "simulation_state.joblib",
            'json_summary': self.working_folder / "summary.json",
            'recovery_marker': self.working_folder / "simulation_in_progress.marker"
        }

    def save_simulation_data(self) -> None:
        """Save current simulation data to disk."""
        if self.working_folder is None:
            return

        paths = self._get_save_paths()

        try:
            # Save order parameters and fluctuations using OrderParametersHistory methods
            order_params_array = self.local_history._get_order_parameters_array()
            fluctuations_array = self.local_history._get_fluctuations_array()
            self.local_history.save_to_npz(
                order_parameters_path=str(paths['order_parameters']) if len(order_params_array) > 0 else None,
                fluctuations_path=str(paths['fluctuations']) if len(fluctuations_array) > 0 else None
            )

            # Save lattice state using LatticeState method
            lattice_save_data = self.state.to_npz_dict(include_lattice=True, include_parameters=True)
            lattice_save_data['current_step'] = self.current_step
            lattice_save_data['relevant_history_length'] = self._relevant_history_length
            np.savez_compressed(paths['lattice_state'], **lattice_save_data)

            # Save simulation state using joblib
            simulation_state = {
                'cycles': self.cycles,
                'fluctuations_window': self.fluctuations_window,
                'current_step': self.current_step,
                'relevant_history_length': self._relevant_history_length,
                'save_interval': self.save_interval
            }
            joblib.dump(simulation_state, paths['simulation_state'])

            logger.debug(f"Saved simulation data at step {self.current_step}")

        except Exception as e:
            logger.error(f"Failed to save simulation data: {e}")

    def save_json_summary(self) -> None:
        """Save latest order parameters and fluctuations as JSON."""
        if self.working_folder is None:
            return

        paths = self._get_save_paths()

        try:
            summary: Dict[str, Any] = {
                'current_step': self.current_step,
                'total_cycles': self.cycles,
                'parameters': self.state.parameters.to_dict()
            }

            # Add order parameters history data using its to_dict method
            history_data = self.local_history.to_dict()
            summary.update(history_data)

            paths['json_summary'].write_text(json.dumps(summary, indent=2))

            logger.debug(f"Saved JSON summary at step {self.current_step}")

        except Exception as e:
            logger.error(f"Failed to save JSON summary: {e}")

    def attempt_recovery(self) -> bool:
        """Attempt to recover simulation state from working folder."""
        if self.working_folder is None or not self.auto_recover:
            return False

        paths = self._get_save_paths()

        # Check if simulation marker exists (indicates incomplete previous run)
        if not paths['recovery_marker'].exists():
            logger.debug("No incomplete simulation marker found, starting fresh simulation")
            return False

        try:
            logger.info("Attempting to recover simulation state...")

            # Load simulation state
            if paths['simulation_state'].exists():
                sim_state = joblib.load(paths['simulation_state'])
                self.current_step = sim_state.get('current_step', 0)
                self._relevant_history_length = sim_state.get('relevant_history_length', 0)
                logger.debug(f"Recovered simulation state: step {self.current_step}")

            # Load lattice state using LatticeState method
            if paths['lattice_state'].exists():
                lattice_data = np.load(paths['lattice_state'])
                self.state.from_npz_dict(lattice_data, load_lattice=True, load_parameters=False)
                self.current_step = int(lattice_data['current_step'])
                self._relevant_history_length = int(lattice_data['relevant_history_length'])
                logger.debug("Recovered lattice state")

            # Load order parameters and fluctuations using OrderParametersHistory method
            self.local_history.load_from_npz(
                order_parameters_path=str(paths['order_parameters']) if paths['order_parameters'].exists() else None,
                fluctuations_path=str(paths['fluctuations']) if paths['fluctuations'].exists() else None
            )

            if len(self.local_history.order_parameters_list) > 0:
                logger.debug(f"Recovered {len(self.local_history.order_parameters_list)} order parameter entries")
            if len(self.local_history.fluctuations_list) > 0:
                logger.debug(f"Recovered {len(self.local_history.fluctuations_list)} fluctuation entries")

            logger.info(f"Successfully recovered simulation from step {self.current_step}")
            return True

        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            logger.info("Starting fresh simulation")
            return False

    def _create_recovery_marker(self) -> None:
        """Create marker file to indicate simulation is in progress."""
        if self.working_folder is None:
            return

        paths = self._get_save_paths()
        try:
            marker_content = f"Simulation started at step {self.current_step}\nTotal cycles: {self.cycles}\n"
            paths['recovery_marker'].write_text(marker_content)
            logger.debug("Created simulation progress marker")
        except Exception as e:
            logger.error(f"Failed to create simulation progress marker: {e}")

    def _remove_recovery_marker(self) -> None:
        """Remove simulation progress marker file to indicate successful completion."""
        if self.working_folder is None:
            return

        paths = self._get_save_paths()
        try:
            if paths['recovery_marker'].exists():
                paths['recovery_marker'].unlink()
                logger.debug("Removed simulation progress marker")
        except Exception as e:
            logger.error(f"Failed to remove simulation progress marker: {e}")


class OrderParametersSaver(Updater):
    """Updater that saves order parameters history periodically."""

    def __init__(self, order_parameters_history: OrderParametersHistory,
                 save_path: pathlib.Path, how_often: int = 1000, since_when: int = 0):
        super().__init__(how_often, since_when)
        self.order_parameters_history = order_parameters_history
        self.save_path = save_path

    def update(self, state: LatticeState):
        """Save order parameters history."""
        try:
            if len(self.order_parameters_history.order_parameters_list) > 0:
                self.order_parameters_history.save_to_npz(order_parameters_path=str(self.save_path))
                logger.debug(f"Saved order parameters to {self.save_path}")
            return f"Order parameters saved at step {state.iterations}"
        except Exception as e:
            logger.error(f"Failed to save order parameters: {e}")
            return f"Save failed at step {state.iterations}"


class FluctuationsSaver(Updater):
    """Updater that saves fluctuations history periodically."""

    def __init__(self, order_parameters_history: OrderParametersHistory,
                 save_path: pathlib.Path, how_often: int = 1000, since_when: int = 0):
        super().__init__(how_often, since_when)
        self.order_parameters_history = order_parameters_history
        self.save_path = save_path

    def update(self, state: LatticeState):
        """Save fluctuations history."""
        try:
            if len(self.order_parameters_history.fluctuations_list) > 0:
                self.order_parameters_history.save_to_npz(fluctuations_path=str(self.save_path))
                logger.debug(f"Saved fluctuations to {self.save_path}")
            return f"Fluctuations saved at step {state.iterations}"
        except Exception as e:
            logger.error(f"Failed to save fluctuations: {e}")
            return f"Save failed at step {state.iterations}"


class LatticeStateSaver(Updater):
    """Updater that saves lattice state and simulation metadata periodically."""

    def __init__(self, save_path: pathlib.Path, simulation: 'Simulation',
                 how_often: int = 1000, since_when: int = 0):
        super().__init__(how_often, since_when)
        self.save_path = save_path
        self.simulation = simulation

    def update(self, state: LatticeState):
        """Save lattice state and metadata."""
        try:
            lattice_save_data = state.to_npz_dict(include_lattice=True, include_parameters=True)
            lattice_save_data['current_step'] = self.simulation.current_step
            lattice_save_data['relevant_history_length'] = self.simulation._relevant_history_length
            np.savez_compressed(self.save_path, **lattice_save_data)
            logger.debug(f"Saved lattice state to {self.save_path}")
            return f"Lattice state saved at step {state.iterations}"
        except Exception as e:
            logger.error(f"Failed to save lattice state: {e}")
            return f"Save failed at step {state.iterations}"


class SimulationStateSaver(Updater):
    """Updater that saves simulation state using joblib."""

    def __init__(self, save_path: pathlib.Path, simulation: 'Simulation',
                 how_often: int = 1000, since_when: int = 0):
        super().__init__(how_often, since_when)
        self.save_path = save_path
        self.simulation = simulation

    def update(self, state: LatticeState):
        """Save simulation state."""
        try:
            simulation_state = {
                'cycles': self.simulation.cycles,
                'fluctuations_window': self.simulation.fluctuations_window,
                'save_interval': self.simulation.save_interval,
                'current_step': self.simulation.current_step,
                'relevant_history_length': self.simulation._relevant_history_length
            }
            joblib.dump(simulation_state, self.save_path)
            logger.debug(f"Saved simulation state to {self.save_path}")
            return f"Simulation state saved at step {state.iterations}"
        except Exception as e:
            logger.error(f"Failed to save simulation state: {e}")
            return f"Save failed at step {state.iterations}"


class JSONSummarySaver(Updater):
    """Updater that saves JSON summary of latest values periodically."""

    def __init__(self, order_parameters_history: OrderParametersHistory,
                 save_path: pathlib.Path, simulation: 'Simulation',
                 how_often: int = 5000, since_when: int = 0):
        super().__init__(how_often, since_when)
        self.order_parameters_history = order_parameters_history
        self.save_path = save_path
        self.simulation = simulation

    def update(self, state: LatticeState):
        """Save JSON summary."""
        try:
            summary: Dict[str, Any] = {
                'current_step': self.simulation.current_step,
                'total_cycles': self.simulation.cycles,
                'parameters': state.parameters.to_dict()
            }

            # Add order parameters history data using its to_dict method
            history_data = self.order_parameters_history.to_dict()
            summary.update(history_data)

            self.save_path.write_text(json.dumps(summary, indent=2))

            logger.debug(f"Saved JSON summary to {self.save_path}")
            return f"JSON saved at step {state.iterations}"

        except Exception as e:
            logger.error(f"Failed to save JSON summary: {e}")
            return f"JSON save failed at step {state.iterations}"
