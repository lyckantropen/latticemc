from abc import abstractmethod

import numpy as np

from .definitions import (LatticeState, OrderParametersHistory,
                          gathered_order_parameters, simulation_stats)
from .order_parameters import calculate_order_parameters
from .statistical import fluctuation


class Updater:
    """
    A base class for executing code during the execution
    of the simulation. The user can schedule this to be
    run every said number of iterations.
    """

    def __init__(self, how_often, since_when, print_every=None):
        self.how_often = how_often
        self.since_when = since_when
        self.last_value = None
        self.print_every = print_every

    def perform(self, state: LatticeState):
        if state.iterations >= self.since_when and state.iterations % self.how_often == 0:
            self.last_value = self.update(state)
            if self.print_every is not None and state.iterations % self.print_every == 0:
                print(f'[{state.iterations},{state.parameters}]:\t {self.format_value(self.last_value)}')

    def format_value(self, value):
        return str(value)

    @abstractmethod
    def update(self, state: LatticeState):
        pass


class OrderParametersCalculator(Updater):
    def __init__(self, order_parameters_history: OrderParametersHistory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.order_parameters_history = order_parameters_history

    def update(self, state: LatticeState):
        op = calculate_order_parameters(state)
        self.order_parameters_history.order_parameters = np.append(self.order_parameters_history.order_parameters, op)
        self.order_parameters_history.stats = np.append(self.order_parameters_history.stats, np.array([(state.wiggle_rate,)], dtype=simulation_stats))
        return op

    def format_value(self, value):
        s = ','.join([f'{name}={value[name][0]:.5f}' for name in gathered_order_parameters.fields.keys()])
        return 'averg: ' + s


class FluctuationsCalculator(Updater):
    def __init__(self, order_parameters_history: OrderParametersHistory, *args, window=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.order_parameters_history = order_parameters_history

    def update(self, state):
        fluctuations = np.zeros(1, dtype=gathered_order_parameters)
        for name in gathered_order_parameters.fields.keys():
            fluct = state.lattice.particles.size * fluctuation(self.order_parameters_history.order_parameters[name][-100:])
            fluctuations[name] = fluct

        self.order_parameters_history.fluctuations = np.append(self.order_parameters_history.fluctuations, fluctuations)
        return fluctuations

    def format_value(self, value):
        s = ','.join([f'{name}={value[name][0]:.5f}' for name in gathered_order_parameters.fields.keys()])
        return 'fluct: ' + s


class RandomWiggleRateAdjustor(Updater):
    def __init__(self, scale, *args, reset_value=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.reset_value = reset_value

    def update(self, state):
        if self.reset_value is not None:
            state.wiggle_rate = np.random.normal(self.reset_value, scale=self.scale)
        else:
            state.wiggle_rate = np.random.normal(state.wiggle_rate, scale=self.scale)
        return state.wiggle_rate


class DerivativeWiggleRateAdjustor(Updater):
    def __init__(self, order_parameters_history: OrderParametersHistory, how_many: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.how_many = how_many
        self.order_parameters_history = order_parameters_history

    def update(self, state):
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
    def __init__(self, callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = callback

    def update(self, state):
        return self.callback(state)
