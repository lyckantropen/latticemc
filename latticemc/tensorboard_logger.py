"""TensorBoard logging utilities for lattice Monte Carlo simulations."""

import io
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter

from .definitions import DefiningParameters, LatticeState, OrderParametersHistory

matplotlib.use('Agg')  # Use non-interactive backend


logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """
    Logger for TensorBoard visualization of simulation data.

    This class handles logging of order parameters, fluctuations,
    energy, and acceptance rates for visualization in TensorBoard.
    """

    def __init__(self, log_dir: Optional[str] = None):
        """Initialize TensorBoard logger.

        Parameters
        ----------
        log_dir : str, optional
            Directory for TensorBoard logs. If None, creates a directory
            in 'runs/latticemc_TIMESTAMP'.
        """
        # Create log directory with timestamp if not provided
        if log_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = os.path.join('runs', f'latticemc_{timestamp}')

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        self.order_parameter_fields = [
            'energy', 'q0', 'q2', 'w', 'p', 'd322'
        ]

        # Step tracking indexed by DefiningParameters
        self._steps: Dict[DefiningParameters, int] = {}
        self._global_step: int = 0

        logger.info(f"TensorBoard logging to {log_dir}")

    def _get_step(self, parameters: Optional[DefiningParameters] = None) -> int:
        """
        Get current step for the given parameters.

        Parameters
        ----------
        parameters : DefiningParameters, optional
            Parameters to get step for. If None, returns global step.

        Returns
        -------
        int
            Current step for the parameters
        """
        if parameters is None:
            return self._global_step
        return self._steps.get(parameters, 0)

    def _increment_step(self, parameters: Optional[DefiningParameters] = None) -> int:
        """
        Increment and return the step for the given parameters.

        Parameters
        ----------
        parameters : DefiningParameters, optional
            Parameters to increment step for. If None, increments global step.

        Returns
        -------
        int
            New step value after incrementing
        """
        if parameters is None:
            self._global_step += 1
            return self._global_step

        if parameters not in self._steps:
            self._steps[parameters] = 0
        self._steps[parameters] += 1
        return self._steps[parameters]

    def log_single_scalar(self, tag: str, value: float, step: int):
        """
        Log a single scalar value to TensorBoard.

        Parameters
        ----------
        tag : str
            TensorBoard tag for the scalar
        value : float
            Value to log
        step : int
            Step/iteration for TensorBoard
        """
        try:
            self.writer.add_scalar(tag, value, step)
        except Exception as e:
            logger.error(f"Error logging scalar {tag}: {e}")

    def log_temperature_scalar_auto(self, category: str, field: str, parameters: DefiningParameters, value: float, step: Optional[int] = None):
        """
        Log a scalar value for a specific temperature with step from parameter or automatic incrementing.

        Parameters
        ----------
        category : str
            Category (e.g., 'order_parameters', 'fluctuations', 'acceptance_rates')
        field : str
            Field name (e.g., 'energy', 'q0', 'orientation')
        parameters : DefiningParameters
            Parameters defining the temperature and other simulation settings
        value : float
            Value to log
        step : int, optional
            Step/iteration to use. If None, uses automatic step incrementing.
        """
        if step is None:
            step = self._increment_step(parameters)
        temperature = float(parameters.temperature)
        tag = f"{category}/{field}/T_{temperature:.2f}"
        self.log_single_scalar(tag, value, step)

    def log_simulation_temperature_assignments(self, states: List[LatticeState], step: int = 0):
        """
        Log initial temperature assignments for each simulation process.

        Parameters
        ----------
        states : List[Any]
            List of LatticeState objects with temperature assignments
        step : int, optional
            Step/iteration for TensorBoard (default: 0 for initial assignment)
        """
        for i, state in enumerate(states):
            tag = f"simulation_index/temperature/process_{i}"
            temperature = float(state.parameters.temperature)
            self.log_single_scalar(tag, temperature, step)

    def log_simulation_temperature_after_exchange_auto(self, process_index: int, temperature: float, step: Optional[int] = None):
        """
        Log temperature assignment for a simulation process after parallel tempering exchange.

        Parameters
        ----------
        process_index : int
            Index of the simulation process
        temperature : float
            Current temperature of the process
        step : int, optional
            Step/iteration to use. If None, uses automatic step incrementing.
        """
        if step is None:
            step = self._increment_step()
        tag = f"simulation_index/current_temperature/process_{process_index}"
        self.log_single_scalar(tag, temperature, step)

    def log_temperature_series(self, tag: str, temperatures: List[float], values: List[float], step: int):
        """
        Log a series of values vs temperature as a custom plot.

        Parameters
        ----------
        tag : str
            Name for the plot (will be used as the image tag)
        temperatures : List[float]
            Temperature values for x-axis
        values : List[float]
            Corresponding y-values
        step : int
            Step/iteration for TensorBoard
        """
        if not temperatures or not values or len(temperatures) != len(values):
            logger.warning(f"Cannot create temperature plot for {tag}: mismatched or empty data")
            return

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(temperatures, values, 'o-', linewidth=2, markersize=4)
            plt.xlabel('Temperature')
            plt.ylabel(tag.replace('_', ' ').title())
            plt.title(f'{tag.replace("_", " ").title()} vs Temperature (Step {step})')
            plt.grid(True, alpha=0.3)

            # Convert to image and log
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            buf.seek(0)

            # Read as PIL image and convert to numpy
            img = Image.open(buf)
            img_array = np.array(img)

            # TensorBoard expects CHW format (Color, Height, Width)
            if len(img_array.shape) == 3:
                img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW

            self.writer.add_image(tag + '_vs_temperature', img_array, step, dataformats='CHW')
            plt.close()
            buf.close()

        except Exception as e:
            logger.error(f"Error creating temperature plot for {tag}: {e}")

    def _log_field_data_vs_temperature(self,
                                       data_type: str,
                                       order_parameters_history: Dict[DefiningParameters, OrderParametersHistory],
                                       step: int,
                                       recent_points: int = 1000,
                                       decorrelation_interval: int = 10) -> None:
        """
        Log field data vs temperature as custom plots.

        Parameters
        ----------
        data_type : str
            Type of data to plot ('order_parameters' or 'fluctuations')
        order_parameters_history : Dict[DefiningParameters, OrderParametersHistory]
            Dictionary mapping parameters to their order parameter histories
        step : int
            Step/iteration for TensorBoard
        recent_points : int, optional
            Number of recent points to average for each temperature (default: 1000)
        decorrelation_interval : int, optional
            Interval to skip points for decorrelation (default: 10)
        """
        if not order_parameters_history:
            logger.warning(f"Cannot create {data_type} temperature plots: empty history")
            return

        try:
            temperatures = []
            field_data: Dict[str, List[float]] = {field: [] for field in self.order_parameter_fields}

            # Choose the appropriate data source
            data_source = 'order_parameters' if data_type == 'order_parameters' else 'fluctuations'

            for params, history in order_parameters_history.items():
                history_data = getattr(history, data_source)
                if history_data.size > 0:
                    temperatures.append(float(params.temperature))
                    this_recent_points = min(recent_points, len(history_data))
                    # Use the most recent points for averaging
                    recent_data = history_data[-this_recent_points::decorrelation_interval] if len(
                        history_data) >= this_recent_points else history_data

                    for field in self.order_parameter_fields:
                        if field in recent_data.dtype.names:
                            field_data[field].append(np.mean(recent_data[field]))
                        else:
                            field_data[field].append(0.0)  # Default value if field not present

            if temperatures:
                # Sort by temperature for cleaner plots
                sorted_indices = np.argsort(temperatures)
                sorted_temps = [temperatures[i] for i in sorted_indices]

                # Create plots for each field
                prefix = 'order_parameter' if data_type == 'order_parameters' else 'fluctuation'
                for field in self.order_parameter_fields:
                    if any(field_data[field]):  # Only plot if we have non-zero data
                        sorted_values = [field_data[field][i] for i in sorted_indices]
                        self.log_temperature_series(f'{prefix}_{field}', sorted_temps, sorted_values, step)

        except Exception as e:
            logger.error(f"Error creating {data_type} temperature plots: {e}")

    def log_order_parameters_vs_temperature(self,
                                            order_parameters_history: Dict[DefiningParameters, OrderParametersHistory],
                                            step: int,
                                            recent_points: int = 100,
                                            decorrelation_interval: int = 10) -> None:
        """
        Log all order parameters vs temperature as custom plots.

        Parameters
        ----------
        order_parameters_history : Dict[DefiningParameters, OrderParametersHistory]
            Dictionary mapping parameters to their order parameter histories
        step : int
            Step/iteration for TensorBoard
        recent_points : int, optional
            Number of recent points to average for each temperature (default: 100)
        decorrelation_interval : int, optional
            Interval to skip points for decorrelation (default: 10)
        """
        self._log_field_data_vs_temperature('order_parameters', order_parameters_history, step, recent_points, decorrelation_interval)

    def log_fluctuations_vs_temperature(self,
                                        order_parameters_history: Dict[DefiningParameters, OrderParametersHistory],
                                        step: int,
                                        recent_points: int = 1000,
                                        decorrelation_interval: int = 10) -> None:
        """
        Log all fluctuations vs temperature as custom plots.

        Parameters
        ----------
        order_parameters_history : Dict[DefiningParameters, OrderParametersHistory]
            Dictionary mapping parameters to their order parameter histories
        step : int
            Step/iteration for TensorBoard
        recent_points : int, optional
            Number of recent points to average for each temperature (default: 1000)
        decorrelation_interval : int, optional
            Interval to skip points for decorrelation (default: 10)
        """
        self._log_field_data_vs_temperature('fluctuations', order_parameters_history, step, recent_points, decorrelation_interval)

    def log_energy_vs_temperature(self,
                                  order_parameters_history: Dict[DefiningParameters, OrderParametersHistory],
                                  step: int,
                                  recent_points: int = 1000,
                                  decorrelation_interval: int = 10) -> None:
        """
        Log energy vs temperature as a custom plot.

        Parameters
        ----------
        order_parameters_history : Dict[DefiningParameters, OrderParametersHistory]
            Dictionary mapping parameters to their order parameter histories
        step : int
            Step/iteration for TensorBoard
        recent_points : int, optional
            Number of recent points to average for each temperature (default: 1000)
        decorrelation_interval : int, optional
            Interval to skip points for decorrelation (default: 10)
        """
        if not order_parameters_history:
            logger.warning("Cannot create energy temperature plot: empty history")
            return

        try:
            temperatures = []
            energies = []

            for params, history in order_parameters_history.items():
                if history.order_parameters.size > 0:
                    temperatures.append(float(params.temperature))
                    # Use the most recent points for averaging
                    recent_op = history.order_parameters[-recent_points::decorrelation_interval] if len(
                        history.order_parameters) >= recent_points else history.order_parameters
                    energies.append(np.mean(recent_op['energy']))

            if temperatures:
                # Sort by temperature for cleaner plots
                sorted_data = sorted(zip(temperatures, energies))
                sorted_temps, sorted_energies = zip(*sorted_data)

                self.log_temperature_series('energy', list(sorted_temps), list(sorted_energies), step)

        except Exception as e:
            logger.error(f"Error creating energy temperature plot: {e}")

    def log_all_temperature_plots(self,
                                  order_parameters_history: Dict[DefiningParameters, OrderParametersHistory],
                                  step: int,
                                  recent_points: int = 1000,
                                  decorrelation_interval: int = 10) -> None:
        """
        Log all available temperature-based plots (energy, order parameters, fluctuations).

        Parameters
        ----------
        order_parameters_history : Dict[DefiningParameters, OrderParametersHistory]
            Dictionary mapping parameters to their order parameter histories
        step : int
            Step/iteration for TensorBoard
        recent_points : int, optional
            Number of recent points to average for each temperature (default: 100)
        decorrelation_interval : int, optional
            Interval to skip points for decorrelation (default: 10)
        """
        self.log_energy_vs_temperature(order_parameters_history, step, recent_points, decorrelation_interval)
        self.log_order_parameters_vs_temperature(order_parameters_history, step, recent_points, decorrelation_interval)
        self.log_fluctuations_vs_temperature(order_parameters_history, step, recent_points, decorrelation_interval)

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
