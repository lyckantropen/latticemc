"""Base simulation class for lattice Monte Carlo simulations."""

import json
import logging
import pathlib
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .definitions import LatticeState, OrderParametersHistory
from .updaters import OrderParametersCalculator, Updater

logger = logging.getLogger(__name__)


class Simulation:
    """
    Lattice Monte Carlo simulation with automatic persistence.

    Parameters
    ----------
    initial_state : LatticeState
        Initial lattice configuration
    cycles : int
        Number of Monte Carlo steps to run
    per_state_updaters : List[Updater], optional
        Additional updaters executed each step
    progress_bar : Any, optional
        Progress bar object (default: console tqdm)
    working_folder : str, optional
        Directory for saving state and results
    save_interval : int, default=200
        Save frequency in steps
    auto_recover : bool, default=False
        Resume from saved state if available
    """

    def __init__(self,
                 initial_state: LatticeState,
                 cycles: int,
                 per_state_updaters: Optional[List[Updater]] = None,
                 progress_bar: Optional[Any] = None,
                 working_folder: Optional[str] = None,
                 save_interval: int = 200,
                 auto_recover: bool = False) -> None:
        self.state = initial_state
        self.cycles = cycles
        self.per_state_updaters = per_state_updaters or []
        self.progress_bar = progress_bar
        self.start_time = time.time()

        # Persistence settings
        self.working_folder = pathlib.Path(working_folder) if working_folder else None
        self.save_interval = save_interval
        self.auto_recover = auto_recover

        self.local_history = OrderParametersHistory(self.state.lattice.size)
        self.current_step = 0

        # Set up working folder if requested
        self._setup_working_folder()

        # Attempt recovery if enabled
        if self.auto_recover and not issubclass(type(self), Simulation):  # avoid in derived classes
            self.recover()

    def tag(self) -> str:
        """Generate a tag for logging."""
        return f'Simulation[{self.state.parameters.tag()}, current_step={self.current_step}]'

    def recover(self) -> bool:
        """Attempt to recover simulation from saved state."""
        if self.working_folder is None:
            logger.debug("No working folder specified, cannot recover")
            return False

        paths = self._get_save_paths()

        # Check if simulation marker exists (indicates incomplete previous run)
        if not paths['recovery_marker'].exists():
            logger.debug("No incomplete simulation marker found, no recovery needed")
            return False

        try:
            logger.info("Attempting to recover simulation state...")

            # Load simulation state from JSON summary
            if paths['json_summary'].exists():
                with open(paths['json_summary'], 'r') as f:
                    summary = json.load(f)
                self.current_step = summary.get('current_step', 0)
                logger.debug(f"Recovered simulation state: step {self.current_step}")

            # Load lattice state using LatticeState method
            if paths['lattice_state'].exists():
                lattice_data = np.load(paths['lattice_state'])
                self.state = LatticeState.from_npz_dict(lattice_data)
                logger.debug("Recovered lattice state, parameters, and recomputed lattice averages")

            # Load order parameters and stats using OrderParametersHistory method
            self.local_history.load_from_npz(
                order_parameters_path=str(paths['order_parameters']) if paths['order_parameters'].exists() else None,
                stats_path=str(paths['stats']) if paths['stats'].exists() else None
            )

            if len(self.local_history.order_parameters_list) > 0:
                logger.debug(f"Recovered {len(self.local_history.order_parameters_list)} order parameter entries")
            if len(self.local_history.stats_list) > 0:
                logger.debug(f"Recovered {len(self.local_history.stats_list)} stats entries")

            logger.info(f"Successfully recovered simulation from step {self.current_step}")
            return True

        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            logger.info("Simulation will start fresh")
            # Reset to fresh state on recovery failure
            self.current_step = 0
            self.local_history = OrderParametersHistory(self.state.lattice.size)
            return False

    def _save_complete_state(self) -> None:
        """Save complete simulation state for recovery."""
        if self.working_folder is None:
            return

        paths = self._get_save_paths()

        try:
            # Ensure all directories exist
            for path in paths.values():
                pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Save lattice state with all metadata needed for recovery
            lattice_save_data = self.state.to_npz_dict()
            lattice_save_data['current_step'] = self.current_step
            np.savez_compressed(paths['lattice_state'], **lattice_save_data)

            # Save order parameters and stats, generate fluctuations from history for final save
            final = self.current_step >= self.cycles
            self.local_history.save_to_npz(
                order_parameters_path=str(paths['order_parameters']) if len(self.local_history.order_parameters_list) > 0 else None,
                fluctuations_path=None,  # No longer saving fluctuations to disk
                stats_path=str(paths['stats']) if len(self.local_history.stats_list) > 0 else None,
                fluctuations_from_history=final  # Generate fluctuations from history on final save
            )

            # Save JSON summary
            summary: Dict[str, Any] = {
                'current_step': self.current_step,
                'total_cycles': self.cycles,
                'parameters': self.state.parameters.to_dict(),
                'finished': self.current_step >= self.cycles,
                'running_time_seconds': time.time() - self.start_time
            }

            if final:
                # note that if this was a parallel tempering simulation, these will be based on truncated history
                op = self.local_history.calculate_decorrelated_averages()
                fl_from_hist = self.local_history.calculate_decorrelated_fluctuations()
                summary['final_order_parameters'] = {name: float(op[name].item()) for name in op.dtype.fields.keys()}  # type: ignore[union-attr]
                summary['final_fluctuations_from_history'] = {name: float(fl_from_hist[name].item())
                                                              for name in fl_from_hist.dtype.fields.keys()}  # type: ignore[union-attr]

            # Add order parameters history data
            history_data = self.local_history.to_dict()
            summary.update(history_data)

            paths['json_summary'].write_text(json.dumps(summary, indent=2))

            logger.debug("Complete simulation state and JSON summary saved successfully")

        except Exception as e:
            logger.error(f"Failed to save complete state: {e}")
            raise

    def run(self) -> None:
        """Execute the simulation."""
        # Create marker to indicate simulation is in progress
        self._create_recovery_marker()

        logger.info(f"Starting simulation with {self.cycles} cycles")
        if self.current_step > 0:
            logger.info(f"Resuming from step {self.current_step}")
        logger.info(f"Parameters: T={self.state.parameters.temperature}, "
                    f"λ={self.state.parameters.lam}, τ={self.state.parameters.tau}")
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
            logger.debug(f'Remaining cycles: {remaining_cycles}')

            for step in progress_iterator:
                # Calculate absolute step number
                absolute_step = start_step + step

                simulation_numba.do_lattice_state_update(self.state)
                self.current_step = absolute_step + 1

                for updater in updaters:
                    updater.perform(self.state)

                # Update progress bar if available
                if self.progress_bar is not None:
                    # update tqdm object by 1 step
                    self.progress_bar.update(1)

        except Exception as e:
            logger.error(f"Simulation failed at step {self.current_step}: {e}")
            self._handle_simulation_error(e)

        self._simulation_finished()

    def _create_progress_iterator(self, remaining_cycles: Optional[int] = None) -> Any:
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
            self.local_history, how_often=1, since_when=0  # this has to be every step
        )
        logger.debug("Added OrderParametersCalculator")

        # Combine with user-provided updaters
        updaters = [
            order_parameters_calculator,
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
            # Complete state saver - saves all data needed for recovery
            complete_saver = CompleteStateSaver(
                self,
                self.local_history,
                how_often=self.save_interval,
                since_when=self.save_interval
            )
            additional_updaters.append(complete_saver)

            logger.debug(f"Added {len(additional_updaters)} persistence updaters")

        return additional_updaters

    def _handle_simulation_error(self, error: Exception) -> None:
        """
        Handle errors that occur during simulation.

        Args
        ----
            error: The exception that occurred during simulation.
        """
        logger.error(f"Simulation error at cycle {self.current_step}: {error}")
        logger.info("Simulation state is saved periodically - check working folder for latest state")

    def _simulation_finished(self) -> None:
        """
        Do stuff when simulation completes successfully.

        Derived classes can override this to perform cleanup or final actions.
        """
        logger.info(f"Simulation completed successfully after {self.cycles} cycles")
        if len(self.local_history.order_parameters_list) > 0:
            final_energy = self.local_history.order_parameters_list[-1]['energy']
            logger.debug(f"Final energy: {final_energy:.6f}")
        logger.debug(f"Total collected data points: {len(self.local_history.order_parameters_list)}")

        # Final save and cleanup if working folder is specified
        if self.working_folder is not None:
            logger.info("Performing final save and cleanup")
            self._save_complete_state()
            self._remove_recovery_marker()
            logger.info("Final save completed")

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
            'stats': self.working_folder / "data" / "stats.npz",
            'lattice_state': self.working_folder / "states" / "lattice_state.npz",
            'json_summary': self.working_folder / "summary.json",
            'recovery_marker': self.working_folder / "simulation_in_progress.marker"
        }

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


class CompleteStateSaver(Updater):
    """Unified saver that saves all data needed for complete simulation recovery."""

    def __init__(self, simulation: 'Simulation', order_parameters_history: OrderParametersHistory,
                 how_often: int = 1000, since_when: int = 0):
        super().__init__(how_often, since_when)
        self.simulation = simulation
        self.order_parameters_history = order_parameters_history

    def update(self, state: LatticeState):
        """Save complete state for recovery."""
        try:
            self.simulation._save_complete_state()
            return f"Complete state saved at step {self.simulation.current_step}"
        except Exception as e:
            logger.error(f"Failed to save complete state: {e}")
            return f"Complete state save failed at step {self.simulation.current_step}"
