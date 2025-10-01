"""Test suite for updaters module."""

from unittest.mock import Mock

import numpy as np
import pytest

from latticemc.definitions import LatticeState, OrderParametersHistory
from latticemc.updaters import (AcceptanceRateWiggleRateAdjustor, CallbackUpdater, FluctuationsCalculator, OrderParametersCalculator, RandomWiggleRateAdjustor,
                                Updater)


class ConcreteUpdater(Updater):
    """Concrete implementation of abstract Updater for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_called = False
        self.update_count = 0

    def update(self, state: LatticeState):
        """Track calls to update method."""
        self.update_called = True
        self.update_count += 1
        return f"update_{self.update_count}"


def create_mock_lattice_state(iterations=100, wiggle_rate=1.0, accepted_x=50, accepted_p=40, particles_size=100):
    """Create a mock LatticeState for testing."""
    state = Mock(spec=LatticeState)
    state.iterations = iterations
    state.wiggle_rate = wiggle_rate
    state.accepted_x = accepted_x
    state.accepted_p = accepted_p
    state.parameters = {"test": "param"}

    # Mock lattice with particles
    lattice = Mock()
    particles = Mock()
    particles.size = particles_size
    lattice.particles = particles
    state.lattice = lattice

    return state


class TestUpdater:
    """Test the base Updater class."""

    def test_init(self):
        updater = ConcreteUpdater(how_often=10, since_when=50, until=200, print_every=20)
        assert updater.how_often == 10
        assert updater.since_when == 50
        assert updater.until == 200
        assert updater.print_every == 20
        assert updater.last_value is None

    def test_init_minimal(self):
        updater = ConcreteUpdater(how_often=10, since_when=50)
        assert updater.how_often == 10
        assert updater.since_when == 50
        assert updater.until is None
        assert updater.print_every is None

    def test_perform_before_since_when(self):
        updater = ConcreteUpdater(how_often=10, since_when=50)
        state = create_mock_lattice_state(iterations=30)

        updater.perform(state)

        assert not updater.update_called
        assert updater.last_value is None

    def test_perform_after_until(self):
        updater = ConcreteUpdater(how_often=10, since_when=50, until=200)
        state = create_mock_lattice_state(iterations=250)

        updater.perform(state)

        assert not updater.update_called
        assert updater.last_value is None

    def test_perform_correct_interval(self):
        updater = ConcreteUpdater(how_often=10, since_when=50)
        state = create_mock_lattice_state(iterations=60)  # 60 % 10 == 0, 60 >= 50

        updater.perform(state)

        assert updater.update_called
        assert updater.last_value == "update_1"

    def test_perform_wrong_interval(self):
        updater = ConcreteUpdater(how_often=10, since_when=50)
        state = create_mock_lattice_state(iterations=55)  # 55 % 10 != 0

        updater.perform(state)

        assert not updater.update_called
        assert updater.last_value is None

    def test_format_value_default(self):
        updater = ConcreteUpdater(how_often=1, since_when=0)
        assert updater.format_value("test") == "test"
        assert updater.format_value(123) == "123"


class TestAcceptanceRateWiggleRateAdjustor:
    """Test the AcceptanceRateWiggleRateAdjustor class."""

    def test_init_default(self):
        adjuster = AcceptanceRateWiggleRateAdjustor(how_often=1, since_when=0)
        assert adjuster.lower_bound == 0.2
        assert adjuster.upper_bound == 0.5
        assert adjuster.discount_factor == 0.95
        assert adjuster.adjustment_factor == 0.05
        assert adjuster.smoothed_acceptance_rate is None

    def test_init_custom_bounds(self):
        adjuster = AcceptanceRateWiggleRateAdjustor(
            lower_bound=0.3, upper_bound=0.7,
            discount_factor=0.9, adjustment_factor=0.1,
            how_often=1, since_when=0
        )
        assert adjuster.lower_bound == 0.3
        assert adjuster.upper_bound == 0.7
        assert adjuster.discount_factor == 0.9
        assert adjuster.adjustment_factor == 0.1

    def test_update_initialization(self):
        adjuster = AcceptanceRateWiggleRateAdjustor(how_often=1, since_when=0)
        state = create_mock_lattice_state(wiggle_rate=1.0, accepted_x=30, particles_size=100)

        wiggle_rate = adjuster.update(state)

        assert adjuster.smoothed_acceptance_rate == 0.3  # 30/100
        assert wiggle_rate == state.wiggle_rate

    def test_update_exponential_moving_average(self):
        adjuster = AcceptanceRateWiggleRateAdjustor(discount_factor=0.8, how_often=1, since_when=0)
        state = create_mock_lattice_state(wiggle_rate=1.0, accepted_x=30, particles_size=100)

        # First update - initialization
        adjuster.update(state)
        assert adjuster.smoothed_acceptance_rate == 0.3

        # Second update - should use exponential moving average
        state.accepted_x = 50  # New acceptance rate: 0.5
        adjuster.update(state)
        expected = 0.8 * 0.3 + 0.2 * 0.5  # 0.24 + 0.1 = 0.34
        assert np.isclose(adjuster.smoothed_acceptance_rate, expected, rtol=1e-10)

    def test_update_high_acceptance_rate(self):
        adjuster = AcceptanceRateWiggleRateAdjustor(
            lower_bound=0.2, upper_bound=0.5, adjustment_factor=0.1,
            how_often=1, since_when=0
        )
        state = create_mock_lattice_state(wiggle_rate=1.0, accepted_x=70, particles_size=100)

        wiggle_rate = adjuster.update(state)

        # Acceptance rate = 0.7, which is > upper_bound (0.5)
        # Should increase wiggle rate
        assert wiggle_rate > 1.0
        assert state.wiggle_rate == wiggle_rate

    def test_update_low_acceptance_rate(self):
        adjuster = AcceptanceRateWiggleRateAdjustor(
            lower_bound=0.2, upper_bound=0.5, adjustment_factor=0.1,
            how_often=1, since_when=0
        )
        state = create_mock_lattice_state(wiggle_rate=1.0, accepted_x=10, particles_size=100)

        wiggle_rate = adjuster.update(state)

        # Acceptance rate = 0.1, which is < lower_bound (0.2)
        # Should decrease wiggle rate
        assert wiggle_rate < 1.0
        assert state.wiggle_rate == wiggle_rate

    def test_update_within_bounds(self):
        adjuster = AcceptanceRateWiggleRateAdjustor(
            lower_bound=0.2, upper_bound=0.5,
            how_often=1, since_when=0
        )
        state = create_mock_lattice_state(wiggle_rate=1.0, accepted_x=30, particles_size=100)

        wiggle_rate = adjuster.update(state)

        # Acceptance rate = 0.3, which is within bounds [0.2, 0.5]
        # Should not significantly change wiggle rate
        assert np.isclose(wiggle_rate, 1.0, rtol=1e-3)

    def test_wiggle_rate_bounds_checking(self):
        adjuster = AcceptanceRateWiggleRateAdjustor(
            lower_bound=0.01, upper_bound=0.02, adjustment_factor=1.0,  # Extreme settings
            how_often=1, since_when=0
        )

        # Test upper bound
        state = create_mock_lattice_state(wiggle_rate=5.0, accepted_x=90, particles_size=100)
        wiggle_rate = adjuster.update(state)
        assert wiggle_rate <= 10.0  # Should be capped at 10.0

        # Test lower bound
        state = create_mock_lattice_state(wiggle_rate=0.001, accepted_x=1, particles_size=100)
        wiggle_rate = adjuster.update(state)
        assert wiggle_rate >= 1e-6  # Should be capped at 1e-6

    def test_adjustment_prevents_too_low_wiggle_rate(self):
        adjuster = AcceptanceRateWiggleRateAdjustor(
            lower_bound=0.5, upper_bound=0.8, adjustment_factor=0.5,
            how_often=1, since_when=0
        )
        state = create_mock_lattice_state(wiggle_rate=1.0, accepted_x=10, particles_size=100)

        wiggle_rate = adjuster.update(state)

        # Even with very low acceptance rate, adjustment should not go below 0.1 factor
        # The adjustment factor is max(adjustment, 0.1)
        assert wiggle_rate >= 0.1

    def test_format_value(self):
        adjuster = AcceptanceRateWiggleRateAdjustor(how_often=1, since_when=0)

        # Before any update
        formatted = adjuster.format_value(1.5)
        assert formatted == "wiggle_rate=1.500000"

        # After update with smoothed acceptance rate
        state = create_mock_lattice_state(wiggle_rate=1.0, accepted_x=30, particles_size=100)
        adjuster.update(state)
        formatted = adjuster.format_value(1.5)
        assert "wiggle_rate=1.500000" in formatted
        assert "smoothed_acc_rate=0.300" in formatted

    def test_multiple_updates_sliding_window(self):
        adjuster = AcceptanceRateWiggleRateAdjustor(
            discount_factor=0.9, adjustment_factor=0.1,
            how_often=1, since_when=0
        )
        state = create_mock_lattice_state(wiggle_rate=1.0, accepted_x=20, particles_size=100)

        # Series of updates to test sliding window behavior
        acceptance_rates = [0.2, 0.3, 0.4, 0.5, 0.6]
        smoothed_rates = []

        for i, rate in enumerate(acceptance_rates):
            state.accepted_x = int(rate * 100)
            adjuster.update(state)
            smoothed_rates.append(adjuster.smoothed_acceptance_rate)

        # First rate should be exact
        assert smoothed_rates[0] == 0.2

        # Subsequent rates should show exponential smoothing
        # Each new rate should have less influence than the previous smoothed value
        for i in range(1, len(smoothed_rates)):
            # The smoothed rate should be between the previous smoothed rate and current rate
            current_rate = acceptance_rates[i]
            prev_smoothed = smoothed_rates[i - 1]
            current_smoothed = smoothed_rates[i]

            if current_rate > prev_smoothed:
                assert prev_smoothed < current_smoothed < current_rate
            else:
                assert current_rate < current_smoothed < prev_smoothed


class TestRandomWiggleRateAdjustor:
    """Test the RandomWiggleRateAdjustor class."""

    def test_init(self):
        adjuster = RandomWiggleRateAdjustor(scale=0.1, how_often=1, since_when=0)
        assert adjuster.scale == 0.1
        assert adjuster.reset_value is None

    def test_init_with_reset_value(self):
        adjuster = RandomWiggleRateAdjustor(scale=0.1, reset_value=2.0, how_often=1, since_when=0)
        assert adjuster.scale == 0.1
        assert adjuster.reset_value == 2.0

    def test_update_without_reset_value(self):
        np.random.seed(42)  # For reproducible test
        adjuster = RandomWiggleRateAdjustor(scale=0.1, how_often=1, since_when=0)
        state = create_mock_lattice_state(wiggle_rate=1.0)

        original_wiggle_rate = state.wiggle_rate
        wiggle_rate = adjuster.update(state)

        # Should have changed the wiggle rate
        assert wiggle_rate != original_wiggle_rate
        assert state.wiggle_rate == wiggle_rate

    def test_update_with_reset_value(self):
        np.random.seed(42)  # For reproducible test
        adjuster = RandomWiggleRateAdjustor(scale=0.1, reset_value=2.0, how_often=1, since_when=0)
        state = create_mock_lattice_state(wiggle_rate=1.0)

        wiggle_rate = adjuster.update(state)

        # Should be centered around reset_value, not original wiggle_rate
        assert abs(wiggle_rate - 2.0) < abs(wiggle_rate - 1.0)
        assert state.wiggle_rate == wiggle_rate


class TestCallbackUpdater:
    """Test the CallbackUpdater class."""

    def test_init(self):
        callback = Mock()
        updater = CallbackUpdater(callback, how_often=1, since_when=0)
        assert updater.callback is callback

    def test_update_calls_callback(self):
        callback = Mock(return_value="callback_result")
        updater = CallbackUpdater(callback, how_often=1, since_when=0)
        state = create_mock_lattice_state()

        result = updater.update(state)

        callback.assert_called_once_with(state)
        assert result == "callback_result"


class TestOrderParametersCalculator:
    """Test the OrderParametersCalculator class."""

    def test_init(self):
        history = Mock(spec=OrderParametersHistory)
        calculator = OrderParametersCalculator(history, how_often=1, since_when=0)
        assert calculator.order_parameters_history is history


class TestFluctuationsCalculator:
    """Test the FluctuationsCalculator class."""

    def test_init(self):
        history = Mock(spec=OrderParametersHistory)
        calculator = FluctuationsCalculator(
            history, how_often=1, since_when=0,
            window=500, decorrelation_interval=5
        )
        assert calculator.order_parameters_history is history
        assert calculator.window == 500
        assert calculator.decorrelation_interval == 5

    def test_init_default_parameters(self):
        history = Mock(spec=OrderParametersHistory)
        calculator = FluctuationsCalculator(history, how_often=1, since_when=0)
        assert calculator.window == 1000
        assert calculator.decorrelation_interval == 10


if __name__ == "__main__":
    pytest.main([__file__])
