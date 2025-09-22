"""Plotting utilities for lattice Monte Carlo simulations."""

import io
import logging
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .definitions import DefiningParameters, OrderParametersHistory

matplotlib.use('Agg')  # Use non-interactive backend

logger = logging.getLogger(__name__)


def create_temperature_series_plot(tag: str, temperatures: List[float], values: List[float]) -> Optional[Image.Image]:
    """
    Create a plot of values vs temperature and return as PIL Image.

    Parameters
    ----------
    tag : str
        Name for the plot title and axis labels
    temperatures : List[float]
        Temperature values for x-axis
    values : List[float]
        Corresponding y-values

    Returns
    -------
    PIL.Image.Image or None
        Generated plot as PIL Image, or None if creation failed
    """
    if not temperatures or not values or len(temperatures) != len(values):
        logger.warning(f"Cannot create temperature plot for {tag}: mismatched or empty data")
        return None

    try:
        plt.figure(figsize=(10, 6))
        plt.plot(temperatures, values, 'o-', linewidth=2, markersize=4)
        plt.xlabel('Temperature')
        plt.ylabel(tag.replace('_', ' ').title())
        plt.title(f'{tag.replace("_", " ").title()} vs Temperature')
        plt.grid(True, alpha=0.3)

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)

        img = Image.open(buf)
        # Convert to RGB to ensure compatibility
        if img.mode != 'RGB':
            img = img.convert('RGB')  # type: ignore

        plt.close()
        buf.close()

        return img

    except Exception as e:
        logger.error(f"Error creating temperature plot for {tag}: {e}")
        return None


def create_field_data_vs_temperature_plots(
    data_type: str,
    order_parameters_history: Dict[DefiningParameters, OrderParametersHistory],
    recent_points: int = 1000,
    decorrelation_interval: int = 10
) -> Dict[str, Image.Image]:
    """
    Create field data vs temperature plots and return as PIL Images.

    Parameters
    ----------
    data_type : str
        Type of data to plot ('order_parameters' or 'fluctuations')
    order_parameters_history : Dict[DefiningParameters, OrderParametersHistory]
        Dictionary mapping parameters to their order parameter histories
    recent_points : int, optional
        Number of recent points to average for each temperature (default: 1000)
    decorrelation_interval : int, optional
        Interval to skip points for decorrelation (default: 10)

    Returns
    -------
    Dict[str, PIL.Image.Image]
        Dictionary mapping field names to generated plot images
    """
    if not order_parameters_history:
        logger.warning(f"Cannot create {data_type} temperature plots: empty history")
        return {}

    try:
        temperatures = []
        field_data: Dict[str, List[float]] = {}

        # Initialize field data dictionary
        order_parameter_fields = ['energy', 'q0', 'q2', 'w', 'p', 'd322']
        field_data = {field: [] for field in order_parameter_fields}

        for params, history in order_parameters_history.items():
            temperatures.append(float(params.temperature))

            # Use the calculate_decorrelated_averages method
            decorrelated_op, decorrelated_fl = history.calculate_decorrelated_averages(
                limit_history=recent_points,
                decorrelation_interval=decorrelation_interval
            )

            # Choose the appropriate data source
            source_data = decorrelated_op if data_type == 'order_parameters' else decorrelated_fl

            for field in order_parameter_fields:
                if field in source_data.dtype.fields.keys():
                    field_data[field].append(float(source_data[field][0]))
                else:
                    field_data[field].append(0.0)  # Default value if field not present

        if not temperatures:
            return {}

        # Sort by temperature for cleaner plots
        sorted_indices = np.argsort(temperatures)
        sorted_temps = [temperatures[i] for i in sorted_indices]

        # Create plots for each field that has non-zero data
        plots = {}
        prefix = 'order_parameter' if data_type == 'order_parameters' else 'fluctuation'

        for field in order_parameter_fields:
            if any(field_data[field]):  # Only plot if we have non-zero data
                sorted_values = [field_data[field][i] for i in sorted_indices]
                plot_tag = f'{prefix}_{field}'
                img = create_temperature_series_plot(plot_tag, sorted_temps, sorted_values)
                if img:
                    plots[field] = img

        return plots

    except Exception as e:
        logger.error(f"Error creating {data_type} temperature plots: {e}")
        return {}


def create_order_parameters_vs_temperature_plots(
    order_parameters_history: Dict[DefiningParameters, OrderParametersHistory],
    recent_points: int = 1000,
    decorrelation_interval: int = 10
) -> Dict[str, Image.Image]:
    """
    Create all order parameters vs temperature plots and return as PIL Images.

    Parameters
    ----------
    order_parameters_history : Dict[DefiningParameters, OrderParametersHistory]
        Dictionary mapping parameters to their order parameter histories
    recent_points : int, optional
        Number of recent points to average for each temperature (default: 1000)
    decorrelation_interval : int, optional
        Interval to skip points for decorrelation (default: 10)

    Returns
    -------
    Dict[str, PIL.Image.Image]
        Dictionary mapping field names to generated plot images
    """
    return create_field_data_vs_temperature_plots(
        'order_parameters', order_parameters_history, recent_points, decorrelation_interval
    )


def create_fluctuations_vs_temperature_plots(
    order_parameters_history: Dict[DefiningParameters, OrderParametersHistory],
    recent_points: int = 1000,
    decorrelation_interval: int = 10
) -> Dict[str, Image.Image]:
    """
    Create all fluctuations vs temperature plots and return as PIL Images.

    Parameters
    ----------
    order_parameters_history : Dict[DefiningParameters, OrderParametersHistory]
        Dictionary mapping parameters to their order parameter histories
    recent_points : int, optional
        Number of recent points to average for each temperature (default: 1000)
    decorrelation_interval : int, optional
        Interval to skip points for decorrelation (default: 10)

    Returns
    -------
    Dict[str, PIL.Image.Image]
        Dictionary mapping field names to generated plot images
    """
    return create_field_data_vs_temperature_plots(
        'fluctuations', order_parameters_history, recent_points, decorrelation_interval
    )


def create_energy_vs_temperature_plot(
    order_parameters_history: Dict[DefiningParameters, OrderParametersHistory],
    recent_points: int = 1000,
    decorrelation_interval: int = 10
) -> Optional[Image.Image]:
    """
    Create energy vs temperature plot and return as PIL Image.

    Parameters
    ----------
    order_parameters_history : Dict[DefiningParameters, OrderParametersHistory]
        Dictionary mapping parameters to their order parameter histories
    recent_points : int, optional
        Number of recent points to average for each temperature (default: 1000)
    decorrelation_interval : int, optional
        Interval to skip points for decorrelation (default: 10)

    Returns
    -------
    PIL.Image.Image or None
        Generated plot as PIL Image, or None if creation failed
    """
    if not order_parameters_history:
        logger.warning("Cannot create energy temperature plot: empty history")
        return None

    try:
        temperatures = []
        energies = []

        for params, history in order_parameters_history.items():
            if len(history.order_parameters_list) > 0:
                temperatures.append(float(params.temperature))

                # Use the calculate_decorrelated_averages method
                decorrelated_op, _ = history.calculate_decorrelated_averages(
                    limit_history=recent_points,
                    decorrelation_interval=decorrelation_interval
                )

                energies.append(float(decorrelated_op['energy'][0]))

        if not temperatures:
            return None

        # Sort by temperature for cleaner plots
        sorted_data = sorted(zip(temperatures, energies))
        sorted_temps, sorted_energies = zip(*sorted_data)

        return create_temperature_series_plot('energy', list(sorted_temps), list(sorted_energies))

    except Exception as e:
        logger.error(f"Error creating energy temperature plot: {e}")
        return None


def create_all_temperature_plots(
    order_parameters_history: Dict[DefiningParameters, OrderParametersHistory],
    recent_points: int = 1000,
    decorrelation_interval: int = 10
) -> Dict[str, Image.Image]:
    """
    Create all available temperature-based plots and return as PIL Images.

    Parameters
    ----------
    order_parameters_history : Dict[DefiningParameters, OrderParametersHistory]
        Dictionary mapping parameters to their order parameter histories
    recent_points : int, optional
        Number of recent points to average for each temperature (default: 1000)
    decorrelation_interval : int, optional
        Interval to skip points for decorrelation (default: 10)

    Returns
    -------
    Dict[str, PIL.Image.Image]
        Dictionary mapping plot names to generated images
    """
    all_plots = {}

    # Energy plot
    energy_plot = create_energy_vs_temperature_plot(
        order_parameters_history, recent_points, decorrelation_interval
    )
    if energy_plot:
        all_plots['energy'] = energy_plot

    # Order parameter plots
    op_plots = create_order_parameters_vs_temperature_plots(
        order_parameters_history, recent_points, decorrelation_interval
    )
    for field, plot in op_plots.items():
        all_plots[f'order_parameter_{field}'] = plot

    # Fluctuation plots
    fluc_plots = create_fluctuations_vs_temperature_plots(
        order_parameters_history, recent_points, decorrelation_interval
    )
    for field, plot in fluc_plots.items():
        all_plots[f'fluctuation_{field}'] = plot

    return all_plots
