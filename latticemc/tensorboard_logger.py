"""TensorBoard logging utilities for lattice Monte Carlo simulations."""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from tensorboardX import SummaryWriter

from .definitions import DefiningParameters, LatticeState
from .plot_utils import create_temperature_series_plot

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
        try:
            img = create_temperature_series_plot(tag, temperatures, values)
            if img:
                # Convert PIL image to numpy array for TensorBoard
                img_array = np.array(img)
                # TensorBoard expects CHW format (Color, Height, Width)
                if len(img_array.shape) == 3:
                    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
                self.writer.add_image(tag + '_vs_temperature', img_array, step, dataformats='CHW')
        except Exception as e:
            logger.error(f"Error logging temperature plot for {tag}: {e}")

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
