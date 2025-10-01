"""Test JSON and NPZ functionality for basic data structure definitions."""

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
from pyfakefs.fake_filesystem_unittest import TestCase

from latticemc.definitions import (DefiningParameters, Lattice, LatticeState, OrderParametersHistory, extract_scalar_value, gathered_order_parameters,
                                   particle_props)
from latticemc.lattice_tools import initialize_partially_ordered
from latticemc.random_quaternion import random_quaternion


class TestDefinitionsJSONSerialization(TestCase):
    """Test JSON serialization methods for basic data structures."""

    def setUp(self):
        """Set up test data structures."""
        self.setUpPyfakefs()

        # Create test parameters
        self.params = DefiningParameters(
            temperature=Decimal("0.85"),
            lam=Decimal("0.35"),
            tau=Decimal("1.25")
        )

        # Create test lattice
        self.lattice = Lattice(4, 4, 4)
        initialize_partially_ordered(self.lattice, x=random_quaternion(1.0))

        # Create test lattice state
        self.state = LatticeState(parameters=self.params, lattice=self.lattice)
        self.state.iterations = 150
        self.state.accepted_x = 75
        self.state.accepted_p = 25

        # Add some lattice averages data
        sample_averages = np.zeros(1, dtype=particle_props)
        sample_averages[0]['energy'] = -4.2
        sample_averages[0]['p'] = 0.75
        sample_averages[0]['t32'] = np.random.random(10).astype(np.float32)
        sample_averages[0]['t20'] = np.random.random(6).astype(np.float32)
        sample_averages[0]['t22'] = np.random.random(6).astype(np.float32)
        sample_averages[0]['index'] = np.array([1, 2, 3], dtype=np.uint16)
        self.state.lattice_averages = sample_averages

        # Create test order parameters history
        self.history = OrderParametersHistory(self.lattice.size)
        sample_ops = np.array([
            (1.5, 0.8, 0.3, 0.5, 0.7, 0.9),
            (1.6, 0.82, 0.32, 0.52, 0.72, 0.92),
            (1.4, 0.78, 0.28, 0.48, 0.68, 0.88)
        ], dtype=gathered_order_parameters)
        self.history.order_parameters = sample_ops

        sample_flucts = np.array([
            (0.1, 0.05, 0.02, 0.03, 0.04, 0.06),
            (0.12, 0.06, 0.03, 0.04, 0.05, 0.07)
        ], dtype=gathered_order_parameters)
        self.history.fluctuations = sample_flucts

    def test_defining_parameters_to_dict(self):
        """Test DefiningParameters.to_dict() method."""
        result = self.params.to_dict()

        # Check structure
        expected_keys = {'parameters_temperature', 'parameters_lam', 'parameters_tau'}
        self.assertEqual(set(result.keys()), expected_keys)

        # Check values and types - should now be strings to preserve Decimal precision
        self.assertEqual(result['parameters_temperature'], '0.85')
        self.assertEqual(result['parameters_lam'], '0.35')
        self.assertEqual(result['parameters_tau'], '1.25')
        self.assertIsInstance(result['parameters_temperature'], str)
        self.assertIsInstance(result['parameters_lam'], str)
        self.assertIsInstance(result['parameters_tau'], str)

    def test_defining_parameters_round_trip_serialization(self):
        """Test round-trip serialization of DefiningParameters maintains exact Decimal precision."""
        # Test various Decimal values that could lose precision with float conversion
        test_cases = [
            # Standard case
            DefiningParameters(temperature=Decimal('1.6'), lam=Decimal('0.3'), tau=Decimal('1.0')),
            # High precision case
            DefiningParameters(temperature=Decimal('1.234567890'), lam=Decimal('0.987654321'), tau=Decimal('2.111111111')),
            # Edge cases
            DefiningParameters(temperature=Decimal('0.1'), lam=Decimal('0.2'), tau=Decimal('0.9')),
            DefiningParameters(temperature=Decimal('10.0'), lam=Decimal('5.5'), tau=Decimal('3.33')),
        ]

        for i, original_params in enumerate(test_cases):
            with self.subTest(case=i, params=original_params):
                # Test direct dict round-trip
                params_dict = original_params.to_dict()
                recovered_params = DefiningParameters.from_dict(params_dict)

                # Check exact equality
                self.assertEqual(original_params, recovered_params,
                                 f"Direct round-trip failed for {original_params}")
                self.assertEqual(hash(original_params), hash(recovered_params),
                                 f"Hash mismatch after direct round-trip for {original_params}")

                # Test JSON round-trip (most critical for recovery)
                json_str = json.dumps(params_dict)
                json_dict = json.loads(json_str)
                json_recovered_params = DefiningParameters.from_dict(json_dict)

                # Check exact equality after JSON serialization
                self.assertEqual(original_params, json_recovered_params,
                                 f"JSON round-trip failed for {original_params}")
                self.assertEqual(hash(original_params), hash(json_recovered_params),
                                 f"Hash mismatch after JSON round-trip for {original_params}")

                # Verify individual Decimal values are preserved exactly
                self.assertEqual(original_params.temperature, json_recovered_params.temperature)
                self.assertEqual(original_params.lam, json_recovered_params.lam)
                self.assertEqual(original_params.tau, json_recovered_params.tau)

    def test_defining_parameters_from_dict_with_numpy_inputs(self):
        """Test DefiningParameters.from_dict() with various numpy input types."""
        # Test with numpy arrays and scalars (like from NPZ files)
        # Note: Some numpy float types may lose precision, so we test the conversion works
        # rather than expecting exact decimal matches
        numpy_inputs = [
            {
                'parameters_temperature': np.float64(1.6),  # float64 preserves this exactly
                'parameters_lam': np.float64(0.3),          # Use float64 for exact preservation
                'parameters_tau': np.int32(1)
            },
            {
                'parameters_temperature': np.array([2.5]),  # 1-element array
                'parameters_lam': np.array([[0.4]]),        # nested array
                'parameters_tau': np.int64(2)
            },
            {
                'parameters_temperature': np.int32(1),      # Integer types are exact
                'parameters_lam': np.int64(0),              # Integer types are exact
                'parameters_tau': np.float64(1.5)          # Use float64 for precision
            }
        ]

        expected_results = [
            DefiningParameters(temperature=Decimal('1.6'), lam=Decimal('0.3'), tau=Decimal('1')),
            DefiningParameters(temperature=Decimal('2.5'), lam=Decimal('0.4'), tau=Decimal('2')),
            DefiningParameters(temperature=Decimal('1'), lam=Decimal('0'), tau=Decimal('1.5'))
        ]

        for i, (numpy_dict, expected) in enumerate(zip(numpy_inputs, expected_results)):
            with self.subTest(case=i, input_dict=numpy_dict):
                result = DefiningParameters.from_dict(numpy_dict)
                self.assertEqual(result, expected,
                                 f"Numpy input conversion failed for case {i}")
                # Test that the result can be used as a dictionary key (proper hashing)
                test_dict = {result: "test_value"}
                self.assertIn(result, test_dict)

    def test_defining_parameters_from_dict_precision_awareness(self):
        """Test that float precision limitations are handled gracefully."""
        # Test cases where numpy float types may lose precision
        float32_dict = {
            'parameters_temperature': np.float32(0.3),  # This will lose precision
            'parameters_lam': np.float32(0.1),
            'parameters_tau': np.float32(1.0)
        }

        # This should still work, just with the precision that float32 provides
        result = DefiningParameters.from_dict(float32_dict)

        # Verify the values are close to expected (but may not be exact due to float32 precision)
        self.assertAlmostEqual(float(result.temperature), 0.3, places=6)
        self.assertAlmostEqual(float(result.lam), 0.1, places=6)
        self.assertAlmostEqual(float(result.tau), 1.0, places=6)

        # Ensure it can still be used as a dictionary key
        test_dict = {result: "test_value"}
        self.assertIn(result, test_dict)

    def test_extract_scalar_value_helper_function(self):
        """Test the extract_scalar_value helper function with various input types."""
        # Test regular Python types (should pass through unchanged)
        self.assertEqual(extract_scalar_value(42), 42)
        self.assertEqual(extract_scalar_value(3.14), 3.14)
        self.assertEqual(extract_scalar_value("test"), "test")
        self.assertEqual(extract_scalar_value([1, 2, 3]), [1, 2, 3])

        # Test numpy scalars (should extract with .item())
        self.assertEqual(extract_scalar_value(np.int32(100)), 100)
        self.assertEqual(extract_scalar_value(np.float64(2.5)), 2.5)
        self.assertEqual(extract_scalar_value(np.float32(1.5)), np.float32(1.5))

        # Test numpy 0-d arrays (should extract with .item())
        self.assertEqual(extract_scalar_value(np.array(5)), 5)
        self.assertEqual(extract_scalar_value(np.array(7.7)), 7.7)

        # Test numpy 1-element arrays (should extract with .item())
        self.assertEqual(extract_scalar_value(np.array([9])), 9)
        self.assertEqual(extract_scalar_value(np.array([4.2])), 4.2)

    def test_lattice_state_to_dict(self):
        """Test LatticeState.to_dict() method."""
        result = self.state.to_dict()

        # Check main structure
        expected_keys = {'simulation_info', 'lattice_averages'}
        self.assertEqual(set(result.keys()), expected_keys)

        # Check simulation info
        sim_info = result['simulation_info']
        expected_sim_keys = {'accepted_x', 'accepted_p', 'lattice_size'}
        self.assertEqual(set(sim_info.keys()), expected_sim_keys)
        self.assertEqual(sim_info['accepted_x'], 75)
        self.assertEqual(sim_info['accepted_p'], 25)
        self.assertEqual(sim_info['lattice_size'], [4, 4, 4])

        # Check lattice averages
        lattice_avgs = result['lattice_averages']
        self.assertIn('energy', lattice_avgs)
        self.assertIn('p', lattice_avgs)
        self.assertIn('t32', lattice_avgs)
        self.assertAlmostEqual(lattice_avgs['energy'], -4.2, places=5)
        self.assertAlmostEqual(lattice_avgs['p'], 0.75, places=5)
        self.assertIsInstance(lattice_avgs['t32'], list)
        self.assertEqual(len(lattice_avgs['t32']), 10)

    def test_order_parameters_history_to_dict(self):
        """Test OrderParametersHistory.to_dict() method."""
        result = self.history.to_dict()

        # Check structure
        expected_keys = {'latest_order_parameters', 'latest_fluctuations', 'data_counts'}
        self.assertEqual(set(result.keys()), expected_keys)

        # Check latest order parameters (should be last entry)
        latest_op = result['latest_order_parameters']
        expected_op_keys = {'energy', 'q0', 'q2', 'w', 'p', 'd322'}
        self.assertEqual(set(latest_op.keys()), expected_op_keys)
        self.assertAlmostEqual(latest_op['energy'], 1.4, places=5)
        self.assertAlmostEqual(latest_op['q0'], 0.78, places=5)

        # Check latest fluctuations
        latest_fluct = result['latest_fluctuations']
        self.assertEqual(set(latest_fluct.keys()), expected_op_keys)
        self.assertAlmostEqual(latest_fluct['energy'], 0.12, places=5)

        # Check data counts
        data_counts = result['data_counts']
        self.assertEqual(data_counts['order_parameters'], 3)
        self.assertEqual(data_counts['fluctuations'], 2)

    def test_json_serialization_complete_workflow(self):
        """Test that all JSON methods produce valid JSON."""
        # Test DefiningParameters
        params_dict = self.params.to_dict()
        params_json = json.dumps(params_dict)
        parsed_params = json.loads(params_json)
        self.assertEqual(parsed_params['parameters_temperature'], '0.85')

        # Test LatticeState
        state_dict = self.state.to_dict()
        state_json = json.dumps(state_dict)
        parsed_state = json.loads(state_json)
        self.assertEqual(parsed_state['simulation_info']['accepted_x'], 75)

        # Test OrderParametersHistory
        history_dict = self.history.to_dict()
        history_json = json.dumps(history_dict)
        parsed_history = json.loads(history_json)
        self.assertEqual(parsed_history['data_counts']['order_parameters'], 3)


class TestDefinitionsNPZSerialization(TestCase):
    """Test NPZ serialization methods for basic data structures."""

    def setUp(self):
        """Set up test data structures."""
        self.setUpPyfakefs()

        # Create test parameters
        self.params = DefiningParameters(
            temperature=Decimal("0.9"),
            lam=Decimal("0.4"),
            tau=Decimal("1.1")
        )

        # Create test lattice
        self.lattice = Lattice(3, 3, 3)
        initialize_partially_ordered(self.lattice, x=random_quaternion(1.0))

        # Create test lattice state
        self.state = LatticeState(parameters=self.params, lattice=self.lattice)
        self.state.iterations = 200
        self.state.accepted_x = 100
        self.state.accepted_p = 50

        # Add lattice averages
        sample_averages = np.zeros(1, dtype=particle_props)
        sample_averages[0]['energy'] = -3.8
        sample_averages[0]['p'] = 0.65
        sample_averages[0]['t32'] = np.arange(10, dtype=np.float32) * 0.1
        sample_averages[0]['t20'] = np.arange(6, dtype=np.float32) * 0.2
        sample_averages[0]['t22'] = np.arange(6, dtype=np.float32) * 0.3
        sample_averages[0]['index'] = np.array([2, 1, 0], dtype=np.uint16)
        self.state.lattice_averages = sample_averages

        # Create test order parameters history
        self.history = OrderParametersHistory(self.lattice.size)
        sample_ops = np.array([
            (2.1, 0.9, 0.4, 0.6, 0.8, 1.0),
            (2.2, 0.91, 0.41, 0.61, 0.81, 1.01)
        ], dtype=gathered_order_parameters)
        self.history.order_parameters = sample_ops

        sample_flucts = np.array([
            (0.2, 0.1, 0.05, 0.06, 0.07, 0.08)
        ], dtype=gathered_order_parameters)
        self.history.fluctuations = sample_flucts

    def test_defining_parameters_to_npz_dict(self):
        """Test DefiningParameters.to_npz_dict() method."""
        result = self.params.to_npz_dict()

        # Check structure and naming convention
        expected_keys = {'parameters_temperature', 'parameters_lam', 'parameters_tau'}
        self.assertEqual(set(result.keys()), expected_keys)

        # Check values and types - now strings to preserve Decimal precision
        self.assertEqual(result['parameters_temperature'], '0.9')
        self.assertEqual(result['parameters_lam'], '0.4')
        self.assertEqual(result['parameters_tau'], '1.1')
        self.assertIsInstance(result['parameters_temperature'], str)

    def test_lattice_to_npz_dict_and_from_npz_dict(self):
        """Test Lattice NPZ serialization and deserialization."""
        # Test serialization
        npz_dict = self.lattice.to_npz_dict()

        expected_keys = {'lattice_X', 'lattice_Y', 'lattice_Z', 'particles', 'properties'}
        self.assertEqual(set(npz_dict.keys()), expected_keys)
        self.assertEqual(npz_dict['lattice_X'], 3)
        self.assertEqual(npz_dict['lattice_Y'], 3)
        self.assertEqual(npz_dict['lattice_Z'], 3)
        self.assertIsNotNone(npz_dict['particles'])
        self.assertIsNotNone(npz_dict['properties'])

        # Test deserialization
        new_lattice = Lattice.from_npz_dict(npz_dict)

        # Check that dimensions were restored
        self.assertEqual(new_lattice.X, 3)
        self.assertEqual(new_lattice.Y, 3)
        self.assertEqual(new_lattice.Z, 3)

        # Check that arrays were restored
        np.testing.assert_array_equal(new_lattice.particles, self.lattice.particles)
        np.testing.assert_array_equal(new_lattice.properties, self.lattice.properties)

    def test_lattice_state_to_npz_dict(self):
        """Test LatticeState.to_npz_dict() method."""
        # Test complete state serialization
        result = self.state.to_npz_dict()

        # Check basic state data
        self.assertEqual(result['iterations'], 200)
        self.assertEqual(result['accepted_x'], 100)
        self.assertEqual(result['accepted_p'], 50)

        # Note: lattice_averages are not saved as they can be recomputed from lattice state
        # Check that lattice_avg_* fields are not present
        lattice_avg_keys = [k for k in result.keys() if k.startswith('lattice_avg_')]
        self.assertEqual(len(lattice_avg_keys), 0, "lattice_avg fields should not be saved")

        # Check lattice data is included
        self.assertIn('lattice_X', result)
        self.assertIn('particles', result)
        self.assertEqual(result['lattice_X'], 3)

        # Check parameters data is included
        self.assertIn('parameters_temperature', result)
        self.assertEqual(result['parameters_temperature'], '0.9')

        # All data should be included (lattice, parameters, and state)
        self.assertIn('lattice_X', result)
        self.assertIn('parameters_temperature', result)
        self.assertIn('iterations', result)

    def test_lattice_state_from_npz_dict(self):
        """Test LatticeState.from_npz_dict() method."""
        # Create NPZ data
        npz_data = self.state.to_npz_dict()

        # Create new state from NPZ data
        new_state = LatticeState.from_npz_dict(npz_data)

        # Check that state data was restored
        self.assertEqual(new_state.iterations, 200)
        self.assertEqual(new_state.accepted_x, 100)
        self.assertEqual(new_state.accepted_p, 50)

        # Check that lattice was restored
        self.assertEqual(new_state.lattice.X, 3)
        self.assertEqual(new_state.lattice.Y, 3)
        self.assertEqual(new_state.lattice.Z, 3)

        # Check that parameters were restored
        self.assertEqual(new_state.parameters.temperature, Decimal("0.9"))
        self.assertEqual(new_state.parameters.lam, Decimal("0.4"))
        self.assertEqual(new_state.parameters.tau, Decimal("1.1"))

    def test_order_parameters_history_save_and_load_npz(self):
        """Test OrderParametersHistory NPZ save and load methods."""
        # Create temporary files using fake filesystem
        op_path = "/test_op.npz"
        fluct_path = "/test_fluct.npz"

        # Test saving
        self.history.save_to_npz(
            order_parameters_path=op_path,
            fluctuations_path=fluct_path
        )

        # Check files were created
        self.assertTrue(Path(op_path).exists())
        self.assertTrue(Path(fluct_path).exists())

        # Test loading
        new_history = OrderParametersHistory(self.lattice.size)
        new_history.load_from_npz(
            order_parameters_path=op_path,
            fluctuations_path=fluct_path
        )

        # Check data was restored correctly
        self.assertEqual(len(new_history.order_parameters), 2)
        self.assertEqual(len(new_history.fluctuations), 1)

        # Check specific values
        np.testing.assert_array_equal(new_history.order_parameters, self.history.order_parameters)
        np.testing.assert_array_equal(new_history.fluctuations, self.history.fluctuations)

        self.assertAlmostEqual(new_history.order_parameters[0]['energy'], 2.1, places=5)
        self.assertAlmostEqual(new_history.fluctuations[0]['energy'], 0.2, places=5)

    def test_order_parameters_history_npz_partial_operations(self):
        """Test OrderParametersHistory NPZ operations with partial data."""
        op_path = "/test_op_only.npz"

        # Test saving only order parameters
        self.history.save_to_npz(order_parameters_path=op_path, fluctuations_path=None)
        self.assertTrue(Path(op_path).exists())

        # Load into new history
        new_history = OrderParametersHistory(self.lattice.size)
        new_history.load_from_npz(order_parameters_path=op_path, fluctuations_path=None)

        # Check that only order parameters were loaded
        self.assertEqual(len(new_history.order_parameters), 2)
        self.assertEqual(len(new_history.fluctuations), 0)  # Should remain empty

    def test_npz_roundtrip_complete_workflow(self):
        """Test complete NPZ workflow with all data structures."""
        # Create comprehensive NPZ data
        full_npz_data = {}

        # Add parameters data
        params_data = self.params.to_npz_dict()
        full_npz_data.update(params_data)

        # Add lattice data
        lattice_data = self.lattice.to_npz_dict()
        full_npz_data.update(lattice_data)

        # Add state data (just the state fields, not duplicating lattice/params)
        state_data = {
            'iterations': self.state.iterations,
            'accepted_x': self.state.accepted_x,
            'accepted_p': self.state.accepted_p
        }
        full_npz_data.update(state_data)

        # Simulate saving and loading via NPZ
        test_path = "/test_complete.npz"
        np.savez_compressed(test_path, **full_npz_data)

        # Load back
        loaded_data = np.load(test_path)

        # Verify all data is present and correct
        self.assertAlmostEqual(float(loaded_data['parameters_temperature']), 0.9, places=5)
        self.assertEqual(int(loaded_data['lattice_X']), 3)
        self.assertEqual(int(loaded_data['iterations']), 200)
        # Note: lattice_avg_* fields are not saved as they can be recomputed

        # Test reconstruction of objects
        new_params = DefiningParameters(
            temperature=Decimal(str(float(loaded_data['parameters_temperature']))),
            lam=Decimal(str(float(loaded_data['parameters_lam']))),
            tau=Decimal(str(float(loaded_data['parameters_tau'])))
        )
        self.assertEqual(new_params.temperature, self.params.temperature)

        loaded_data.close()


if __name__ == '__main__':
    import unittest
    unittest.main()
