"""Update mechanisms for simulation state tracking and analysis."""

from abc import abstractmethod

import numpy as np

from .definitions import LatticeState, OrderParametersHistory, gathered_order_parameters, simulation_stats
from .order_parameters import calculate_order_parameters
from .statistical import fluctuation


class Updater:
    """
    Base class for executing code during execution of the simulation.

    The user can schedule this to be run every defined number of iterations.
    """

    def __init__(self, how_often, since_when, print_every=None):
        self.how_often = how_often
        self.since_when = since_when
        self.last_value = None
        self.print_every = print_every

    def perform(self, state: LatticeState):
        """Execute the update if conditions are met and optionally print results."""
        if state.iterations >= self.since_when and state.iterations % self.how_often == 0:
            self.last_value = self.update(state)
            if self.print_every is not None and state.iterations % self.print_every == 0:
                print(f'[{state.iterations},{state.parameters}]:\t {self.format_value(self.last_value)}')

    def format_value(self, value):
        """Format the update value for display."""
        return str(value)

    @abstractmethod
    def update(self, state: LatticeState):
        """Perform the actual update operation."""
        pass


class OrderParametersCalculator(Updater):
    """Calculate and store order parameters during simulation."""

    def __init__(self, order_parameters_history: OrderParametersHistory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.order_parameters_history = order_parameters_history

    def update(self, state: LatticeState):
        """Calculate and store current order parameters."""
        op = calculate_order_parameters(state)
        self.order_parameters_history.append_order_parameters(op)
        stats_item = np.array([(state.wiggle_rate, state.accepted_x, state.accepted_p)], dtype=simulation_stats)
        self.order_parameters_history.append_stats(stats_item)
        return op

    def format_value(self, value):
        """Format order parameters for display."""
        if gathered_order_parameters.fields is not None:
            s = ','.join([f'{name}={value[name][0]:.5f}' for name in gathered_order_parameters.fields.keys()])
            return 'averg: ' + s
        return 'averg: no fields'


class FluctuationsCalculator(Updater):
    """Calculate fluctuations of order parameters over a sliding window."""

    def __init__(self, order_parameters_history: OrderParametersHistory, *args, window=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.order_parameters_history = order_parameters_history
        self.window = window

    def update(self, state):
        """Calculate fluctuations for recent order parameters."""
        fluctuations = np.zeros(1, dtype=gathered_order_parameters)
        if gathered_order_parameters.fields is not None:
            # Get current order parameters array for fluctuation calculation
            order_params_array = self.order_parameters_history._get_order_parameters_array()
            for name in gathered_order_parameters.fields.keys():
                particles_size = state.lattice.particles.size if state.lattice.particles is not None else 1
                fluct = particles_size * fluctuation(order_params_array[name][-self.window:])
                fluctuations[name] = fluct

        self.order_parameters_history.append_fluctuations(fluctuations)
        return fluctuations

    def format_value(self, value):
        """Format fluctuations for display."""
        if gathered_order_parameters.fields is not None:
            s = ','.join([f'{name}={value[name][0]:.5f}' for name in gathered_order_parameters.fields.keys()])
            return 'fluct: ' + s
        return 'fluct: no fields'


class RandomWiggleRateAdjustor(Updater):
    """Randomly adjust the wiggle rate parameter."""

    def __init__(self, scale, *args, reset_value=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.reset_value = reset_value

    def update(self, state):
        """Randomly adjust wiggle rate using normal distribution."""
        if self.reset_value is not None:
            state.wiggle_rate = np.random.normal(self.reset_value, scale=self.scale)
        else:
            state.wiggle_rate = np.random.normal(state.wiggle_rate, scale=self.scale)
        return state.wiggle_rate


class AcceptanceRateWiggleRateAdjustor(Updater):
    """Adjust wiggle rate based on acceptance rate bounds."""

    def __init__(self, lower_bound: float = 0.2, upper_bound: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def update(self, state: LatticeState) -> float:
        """Adjust wiggle rate based on current acceptance rate."""
        particles_size = state.lattice.particles.size if state.lattice.particles is not None else 1
        acceptance_rate = state.accepted_x / particles_size
        if acceptance_rate > self.upper_bound:
            state.wiggle_rate *= 1.1
        elif acceptance_rate < self.lower_bound:
            state.wiggle_rate *= 0.9
        return state.wiggle_rate


class DerivativeWiggleRateAdjustor(Updater):
    """Adjust wiggle rate based on energy derivative trends."""

    def __init__(self, order_parameters_history: OrderParametersHistory, how_many: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.how_many = how_many
        self.order_parameters_history = order_parameters_history

    def update(self, state):
        """Adjust wiggle rate based on energy derivative analysis."""
        m_e = np.array([m.mean() for m in np.split(self.order_parameters_history.order_parameters['energy'][-self.how_many:], 4)])
        m_r = np.array([m.mean() for m in np.split(self.order_parameters_history.stats['wiggle_rate'][-self.how_many:], 4)])
        m_r[np.where(m_r == 0)] = np.random.normal(scale=0.001)

        de = np.diff(m_e, 1)
        dr = np.diff(m_r, 1)
        efirst = de / dr
        etrend = (efirst[-1] - efirst[-2])
        if etrend < 0:
            state.wiggle_rate *= 1.1
        else:
            state.wiggle_rate *= 0.9
        return state.wiggle_rate


class CallbackUpdater(Updater):
    """Execute a custom callback function during simulation updates."""

    def __init__(self, callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = callback

    def update(self, state):
        """Execute the registered callback function."""
        return self.callback(state)
