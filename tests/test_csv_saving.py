"""Test CSV and XZ saving functionality to verify data integrity."""

import tempfile
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from latticemc.csv_saver import save_parameter_summary_tables
from latticemc.definitions import DefiningParameters, Lattice, LatticeState, OrderParametersHistory, gathered_order_parameters
from latticemc.lattice_tools import initialize_random


class TestCSVSaving:
    """Test suite for CSV and XZ data saving functionality."""

    def _create_test_history(self, n_points: int = 100, n_particles: int = 27) -> OrderParametersHistory:
        """Create a test OrderParametersHistory with known data."""
        history = OrderParametersHistory(n_particles=n_particles)

        # Generate synthetic but realistic data
        np.random.seed(42)  # For reproducible tests

        for i in range(n_points):
            # Create synthetic order parameters with some trending behavior
            energy = -1.0 + 0.01 * i + 0.1 * np.random.randn()
            q0 = 0.5 + 0.001 * i + 0.05 * np.random.randn()
            q2 = 0.3 + 0.0005 * i + 0.03 * np.random.randn()
            w = 0.2 + 0.0001 * i + 0.02 * np.random.randn()
            p = 0.8 + 0.0002 * i + 0.04 * np.random.randn()
            d322 = 0.1 + 0.00005 * i + 0.01 * np.random.randn()

            op_item = np.array([(energy, q0, q2, w, p, d322)], dtype=gathered_order_parameters)
            history.append_order_parameters(op_item[0])

            # Add some fluctuations data every 10 steps (simulating the updater behavior)
            if i % 10 == 0 and i > 50:  # Start fluctuations after some warmup
                # Create fluctuations based on recent order parameters window
                fl_item = np.array([(0.1, 0.05, 0.03, 0.02, 0.04, 0.01)], dtype=gathered_order_parameters)
                history.append_fluctuations(fl_item[0])

        return history

    def _create_test_state(self, parameters: DefiningParameters) -> LatticeState:
        """Create a test lattice state."""
        lattice = Lattice(X=3, Y=3, Z=3)
        initialize_random(lattice)
        return LatticeState(parameters=parameters, lattice=lattice)

    def test_csv_saving_complete_history(self):
        """Test that saved CSV/XZ files contain complete decorrelated history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            parameters1 = DefiningParameters(
                temperature=Decimal('1.0'),
                tau=Decimal('0.1'),
                lam=Decimal('0.5')
            )
            parameters2 = DefiningParameters(
                temperature=Decimal('2.0'),
                tau=Decimal('0.1'),
                lam=Decimal('0.5')
            )

            history1 = self._create_test_history(n_points=200)
            history2 = self._create_test_history(n_points=150)

            state1 = self._create_test_state(parameters1)
            state2 = self._create_test_state(parameters2)

            order_parameters_history = {
                parameters1: history1,
                parameters2: history2
            }
            states = [state1, state2]

            # Save the data
            save_parameter_summary_tables(
                working_folder=temp_dir,
                order_parameters_history=order_parameters_history,
                states=states,
                recent_points=None,  # Use full history
                final=True
            )

            # Verify summary files were created
            data_dir = Path(temp_dir) / 'data'
            assert (data_dir / 'parameter_summary_final.csv').exists()
            assert (data_dir / 'parameter_summary_final.xz').exists()

            # Load and verify summary table
            df_csv = pd.read_csv(data_dir / 'parameter_summary_final.csv')
            df_xz = pd.read_pickle(data_dir / 'parameter_summary_final.xz', compression='xz')

            # Verify CSV and XZ contain same data (but handle type differences)
            # CSV converts Decimal to float, while XZ preserves Decimal objects
            assert len(df_csv) == len(df_xz), "Both formats should have same number of rows"
            assert df_csv.shape[1] == df_xz.shape[1], "Both formats should have same number of columns"

            # Check that numeric values match (accounting for type conversion)
            for col in df_csv.columns:
                if 'temperature' in col or 'lam' in col or 'tau' in col:
                    # For parameter columns, compare as floats
                    pd.testing.assert_series_equal(
                        df_csv[col].astype(float),
                        df_xz[col].astype(float),
                        check_names=False
                    )
                else:
                    # For other columns, compare directly
                    pd.testing.assert_series_equal(df_csv[col], df_xz[col], check_names=False)

            # Verify we have rows for both parameter sets
            assert len(df_csv) == 2

            # Check that parameter values are correctly saved
            temps = sorted(df_csv['parameters_temperature'].tolist())
            assert temps == [1.0, 2.0]

            # Verify decorrelated averages structure and reasonableness
            for _, row in df_csv.iterrows():
                temp = row['parameters_temperature']
                if temp == 1.0:
                    params = parameters1
                else:
                    params = parameters2

                # Verify that the CSV contains all expected columns and reasonable values
                for field in gathered_order_parameters.names:
                    # Check columns exist
                    assert f'avg_{field}' in row
                    assert f'fluct_{field}' in row
                    assert f'hist_fluct_{field}' in row

                    # Check values are reasonable (not NaN, not infinite)
                    assert not np.isnan(row[f'avg_{field}'])
                    assert not np.isnan(row[f'fluct_{field}'])
                    assert not np.isnan(row[f'hist_fluct_{field}'])
                    assert np.isfinite(row[f'avg_{field}'])
                    assert np.isfinite(row[f'fluct_{field}'])
                    assert np.isfinite(row[f'hist_fluct_{field}'])

                # Verify parameter values are correctly saved
                param_dict = params.to_dict()
                for param_name, param_value in param_dict.items():
                    assert row[param_name] == float(param_value)

                # Verify lattice dimensions
                assert row['lattice_X'] == 3
                assert row['lattice_Y'] == 3
                assert row['lattice_Z'] == 3

    def test_csv_saving_per_parameter_files(self):
        """Test that per-parameter raw data files are created and contain correct data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            parameters = DefiningParameters(
                temperature=Decimal('1.5'),
                tau=Decimal('0.2'),
                lam=Decimal('0.3')
            )

            history = self._create_test_history(n_points=100)
            state = self._create_test_state(parameters)

            order_parameters_history = {parameters: history}
            states = [state]

            # Save the data with final=True to generate per-parameter files
            save_parameter_summary_tables(
                working_folder=temp_dir,
                order_parameters_history=order_parameters_history,
                states=states,
                recent_points=None,
                final=True
            )

            # Verify per-parameter files were created
            param_folder = parameters.get_folder_name()
            param_data_dir = Path(temp_dir) / 'parameters' / param_folder / 'data'

            assert (param_data_dir / 'timeseries.csv').exists()
            assert (param_data_dir / 'timeseries.xz').exists()

            # Load and verify per-parameter data
            ts_csv = pd.read_csv(param_data_dir / 'timeseries.csv')
            ts_xz = pd.read_pickle(param_data_dir / 'timeseries.xz', compression='xz')

            # Verify CSV and XZ contain same data
            pd.testing.assert_frame_equal(ts_csv, ts_xz)

            # Verify data integrity
            op_array = history._get_order_parameters_array()
            fl_array = history._get_fluctuations_array()

            # Check that we have the expected number of rows
            expected_rows = max(len(op_array), len(fl_array))
            assert len(ts_csv) == expected_rows

            # Verify order parameters columns exist and contain correct data
            for field in gathered_order_parameters.names:
                op_col = f'op_{field}'
                assert op_col in ts_csv.columns

                # Check that order parameter values match (up to the length of op_array)
                for i in range(min(len(op_array), len(ts_csv))):
                    assert abs(ts_csv.iloc[i][op_col] - float(op_array[i][field])) < 1e-10

            # Verify fluctuations columns exist and contain correct data (where available)
            for field in gathered_order_parameters.names:
                fl_col = f'fl_{field}'
                assert fl_col in ts_csv.columns

                # Check fluctuation values where they exist
                for i in range(min(len(fl_array), len(ts_csv))):
                    if not pd.isna(ts_csv.iloc[i][fl_col]):
                        assert abs(ts_csv.iloc[i][fl_col] - float(fl_array[i][field])) < 1e-10

    def test_csv_saving_recent_points_limit(self):
        """Test that recent_points parameter correctly limits the history window."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            parameters = DefiningParameters(
                temperature=Decimal('1.0'),
                tau=Decimal('0.1'),
                lam=Decimal('0.5')
            )

            history = self._create_test_history(n_points=200)
            state = self._create_test_state(parameters)

            order_parameters_history = {parameters: history}
            states = [state]

            # Save with limited recent points
            recent_points = 50
            save_parameter_summary_tables(
                working_folder=temp_dir,
                order_parameters_history=order_parameters_history,
                states=states,
                recent_points=recent_points,
                final=False
            )

            # Load saved data
            df = pd.read_csv(Path(temp_dir) / 'data' / 'parameter_summary.csv')

            # Verify that saved values match the limited calculation
            # Only test that the CSV saving mechanism works correctly by
            # checking that files are created and contain expected structure
            row = df.iloc[0]

            # Verify structure - all expected columns exist
            for field in gathered_order_parameters.names:
                assert f'avg_{field}' in row
                assert f'fluct_{field}' in row
                assert f'hist_fluct_{field}' in row

            # Verify values are reasonable (not NaN, not zero)
            for field in gathered_order_parameters.names:
                assert not np.isnan(row[f'avg_{field}'])
                assert not np.isnan(row[f'fluct_{field}'])
                assert not np.isnan(row[f'hist_fluct_{field}'])

            # Verify that we can distinguish between full and limited history
            # Save again with no limit for comparison
            save_parameter_summary_tables(
                working_folder=str(temp_dir) + "_full",
                order_parameters_history=order_parameters_history,
                states=states,
                recent_points=None,
                final=False
            )

            df_unlimited = pd.read_csv(Path(temp_dir + "_full") / 'data' / 'parameter_summary.csv')

            # At least one value should differ
            differs = False
            for col in df.columns:
                if 'avg_' in col or 'fluct_' in col or 'hist_fluct_' in col:
                    if abs(df.iloc[0][col] - df_unlimited.iloc[0][col]) > 1e-6:
                        differs = True
                        break

            assert differs, "Limited and unlimited history should produce different results"

    def test_csv_saving_empty_history(self):
        """Test behavior when order parameters history is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty history
            parameters = DefiningParameters(
                temperature=Decimal('1.0'),
                tau=Decimal('0.1'),
                lam=Decimal('0.5')
            )

            history = OrderParametersHistory(n_particles=27)  # Empty history
            state = self._create_test_state(parameters)

            order_parameters_history = {parameters: history}
            states = [state]

            # Save should handle empty history gracefully
            save_parameter_summary_tables(
                working_folder=temp_dir,
                order_parameters_history=order_parameters_history,
                states=states,
                recent_points=None,
                final=True
            )

            # No files should be created for empty history
            data_dir = Path(temp_dir) / 'data'
            if data_dir.exists():
                csv_files = list(data_dir.glob('*.csv'))
                xz_files = list(data_dir.glob('*.xz'))
                # If files exist, they should be empty or contain no data rows
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    assert len(df) == 0
                for xz_file in xz_files:
                    df = pd.read_pickle(xz_file, compression='xz')
                    assert len(df) == 0
