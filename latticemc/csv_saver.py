"""CSV and XZ saving functionality for simulation data.

This module provides functions to save simulation data in CSV and compressed XZ formats
for analysis. The saved tables are organized for easy comparison across parameter sets
and include both summary statistics and raw time-series data.

See SimulationRunner class documentation for complete details of the folder structure
and table formats created during simulation runs.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .definitions import DefiningParameters, LatticeState, OrderParametersHistory

logger = logging.getLogger(__name__)


def save_parameter_summary_tables(working_folder: str,
                                  order_parameters_history: Dict[DefiningParameters, OrderParametersHistory],
                                  states: List[LatticeState],
                                  recent_points: Optional[int] = None,
                                  final: bool = False
                                  ) -> None:
    """Save parameter summary tables as CSV and XZ files.

    Parameters
    ----------
    working_folder : str
        Directory where tables will be saved
    order_parameters_history : Dict[DefiningParameters, OrderParametersHistory]
        Parameter histories to save
    states : List[LatticeState]
        States for extracting lattice dimensions
    recent_points : Optional[int], default None
        Window size for decorrelated averages (None uses full history)
    final : bool, default False
        If True, saves with '_final' suffix and creates per-parameter time series

    Files Created
    -------------
    - data/parameter_summary[_final].csv/.xz: Cross-parameter comparison
    - parameters/*/data/timeseries.csv/.xz: Individual time series (final only)
    """
    try:
        rows = []

        # Process each parameter set
        for parameters, history in order_parameters_history.items():
            if not history or len(history.order_parameters_list) == 0:
                continue

            row = _create_parameter_row(parameters, history, states, recent_points)
            if row:
                rows.append(row)

                # Save per-parameter raw data files if this is final save
                if final:
                    _save_per_parameter_data(working_folder, parameters, history)

        if not rows:
            logger.debug("No parameter histories available to save")
            return

        # Create and save summary table
        df = pd.DataFrame(rows)
        data_dir = Path(working_folder) / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)

        suffix = '_final' if final else ''
        csv_path = data_dir / f'parameter_summary{suffix}.csv'
        xz_path = data_dir / f'parameter_summary{suffix}.xz'

        df.to_csv(csv_path, index=False)
        df.to_pickle(xz_path, compression='xz')

        level = logging.INFO if final else logging.DEBUG
        logger.log(level, f"Saved parameter summary to {data_dir}")

    except Exception as e:
        logger.error(f"Failed to save parameter summary tables: {e}")


def _create_parameter_row(parameters: DefiningParameters,
                          history: OrderParametersHistory,
                          states: List[LatticeState],
                          recent_points: Optional[int]
                          ) -> Optional[Dict[str, Any]]:
    """Create a row of data for the parameter summary table."""
    try:
        # Get lattice dimensions from matching state
        matching_state = next((s for s in states if s.parameters == parameters), None)
        lattice_dims = _get_lattice_dimensions(matching_state)

        # Calculate statistics
        op_avg = history.calculate_decorrelated_averages(limit_history=recent_points)
        fl_values = history.calculate_decorrelated_fluctuations(limit_history=recent_points)

        # Build row dictionary
        row = {**parameters.to_dict(), **lattice_dims}

        # Add averages
        for name in op_avg.dtype.names:  # type: ignore[union-attr]
            row[f'avg_{name}'] = float(op_avg[name].item())

        # Add fluctuations
        for name in fl_values.dtype.names:  # type: ignore[union-attr]
            row[f'fl_{name}'] = float(fl_values[name].item())

        return row

    except Exception as e:
        logger.error(f"Error creating row for parameters {parameters}: {e}")
        return None


def _get_lattice_dimensions(state: Optional["LatticeState"]) -> Dict[str, Optional[int]]:
    """Extract lattice dimensions from state."""
    if state is None:
        return {'lattice_X': None, 'lattice_Y': None, 'lattice_Z': None}

    return {
        'lattice_X': state.lattice.X,
        'lattice_Y': state.lattice.Y,
        'lattice_Z': state.lattice.Z
    }


def _save_per_parameter_data(working_folder: str,
                             parameters: DefiningParameters,
                             history: OrderParametersHistory
                             ) -> None:
    """Save raw time-series data for individual parameter set."""
    try:
        # Create parameter-specific folder
        param_folder = parameters.get_folder_name()
        data_dir = Path(working_folder) / "parameters" / param_folder / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Fluctuations are not saved here because there's no time series for them
        # Combine order parameters into single file
        combined_data = []

        # Get raw arrays
        op_array = history._get_order_parameters_array()

        row: Dict[str, float]
        # Add order parameters with 'op_' prefix
        if len(op_array) > 0:
            for i, item in enumerate(op_array):
                row = {'step': i}
                for name in op_array.dtype.names:  # type: ignore[union-attr]
                    row[f'op_{name}'] = float(item[name])
                combined_data.append(row)

        if combined_data:
            df = pd.DataFrame(combined_data)
            csv_path = data_dir / 'timeseries.csv'
            xz_path = data_dir / 'timeseries.xz'

            df.to_csv(csv_path, index=False)
            df.to_pickle(xz_path, compression='xz')

    except Exception as e:
        logger.error(f"Failed to save per-parameter data for {parameters}: {e}")
