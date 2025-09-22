"""Test JSON and NPZ functionality for basic data structure definitions."""

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
from pyfakefs.fake_filesystem_unittest import TestCase

from latticemc.definitions import DefiningParameters, Lattice, LatticeState, OrderParametersHistory, gathered_order_parameters, particle_props
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
        self.history = OrderParametersHistory()
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
        expected_keys = {'temperature', 'lam', 'tau'}
        self.assertEqual(set(result.keys()), expected_keys)

        # Check values and types
        self.assertEqual(result['temperature'], 0.85)
        self.assertEqual(result['lam'], 0.35)
        self.assertEqual(result['tau'], 1.25)
        self.assertIsInstance(result['temperature'], float)
        self.assertIsInstance(result['lam'], float)
        self.assertIsInstance(result['tau'], float)

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
        self.assertEqual(parsed_params['temperature'], 0.85)

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
        self.history = OrderParametersHistory()
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

        # Check values and types
        self.assertEqual(result['parameters_temperature'], 0.9)
        self.assertEqual(result['parameters_lam'], 0.4)
        self.assertEqual(result['parameters_tau'], 1.1)
        self.assertIsInstance(result['parameters_temperature'], float)

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
        new_lattice = Lattice(1, 1, 1)  # Start with different dimensions
        new_lattice.from_npz_dict(npz_dict)

        # Check that dimensions were restored
        self.assertEqual(new_lattice.X, 3)
        self.assertEqual(new_lattice.Y, 3)
        self.assertEqual(new_lattice.Z, 3)

        # Check that arrays were restored
        np.testing.assert_array_equal(new_lattice.particles, self.lattice.particles)
        np.testing.assert_array_equal(new_lattice.properties, self.lattice.properties)

    def test_lattice_state_to_npz_dict(self):
        """Test LatticeState.to_npz_dict() method."""
        # Test with all options
        result = self.state.to_npz_dict(include_lattice=True, include_parameters=True)

        # Check basic state data
        self.assertEqual(result['iterations'], 200)
        self.assertEqual(result['accepted_x'], 100)
        self.assertEqual(result['accepted_p'], 50)

        # Check lattice averages are included with proper naming
        self.assertIn('lattice_avg_energy', result)
        self.assertIn('lattice_avg_p', result)
        self.assertIn('lattice_avg_t32', result)
        self.assertAlmostEqual(float(result['lattice_avg_energy']), -3.8, places=5)
        np.testing.assert_array_almost_equal(result['lattice_avg_t32'], np.arange(10) * 0.1, decimal=5)

        # Check lattice data is included
        self.assertIn('lattice_X', result)
        self.assertIn('particles', result)
        self.assertEqual(result['lattice_X'], 3)

        # Check parameters data is included
        self.assertIn('parameters_temperature', result)
        self.assertEqual(result['parameters_temperature'], 0.9)

        # Test with selective inclusion
        result_minimal = self.state.to_npz_dict(include_lattice=False, include_parameters=False)
        self.assertNotIn('lattice_X', result_minimal)
        self.assertNotIn('parameters_temperature', result_minimal)
        self.assertIn('iterations', result_minimal)  # State data should still be there

    def test_lattice_state_from_npz_dict(self):
        """Test LatticeState.from_npz_dict() method."""
        # Create NPZ data
        npz_data = self.state.to_npz_dict(include_lattice=True, include_parameters=False)

        # Create new state and restore from NPZ data
        new_params = DefiningParameters(
            temperature=Decimal("0.5"), lam=Decimal("0.1"), tau=Decimal("0.8")
        )
        new_lattice = Lattice(2, 2, 2)
        new_state = LatticeState(parameters=new_params, lattice=new_lattice)

        # Load from NPZ dict
        new_state.from_npz_dict(npz_data, load_lattice=True, load_parameters=False)

        # Check that state data was restored
        self.assertEqual(new_state.iterations, 200)
        self.assertEqual(new_state.accepted_x, 100)
        self.assertEqual(new_state.accepted_p, 50)

        # Check that lattice was restored
        self.assertEqual(new_state.lattice.X, 3)
        self.assertEqual(new_state.lattice.Y, 3)
        self.assertEqual(new_state.lattice.Z, 3)

        # Check that parameters were NOT restored (load_parameters=False)
        self.assertEqual(new_state.parameters.temperature, Decimal("0.5"))

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
        new_history = OrderParametersHistory()
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
        new_history = OrderParametersHistory()
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

        # Add state data
        state_data = self.state.to_npz_dict(include_lattice=False, include_parameters=False)
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
        self.assertAlmostEqual(float(loaded_data['lattice_avg_energy']), -3.8, places=5)

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
