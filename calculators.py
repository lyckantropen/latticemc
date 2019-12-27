from abc import abstractmethod
import numpy as np
from statistical import fluctuation
from orderParameters import calculateOrderParameters
from definitions import gatheredOrderParameters, LatticeState, OrderParametersHistory


class Calculator:
    """
    A base class for executing code during the execution
    of the simulation. The user can schedule this to be
    run every said number of iterations.
    """
    def __init__(self, howOften, sinceWhen, printEvery=None):
        self.howOften = howOften
        self.sinceWhen = sinceWhen
        self.lastValue = None
        self.printEvery = printEvery

    def perform(self, state: LatticeState):
        if state.iterations >= self.sinceWhen and state.iterations % self.howOften == 0:
            self.lastValue = self.calculate(state)
            if self.printEvery is not None and state.iterations % self.printEvery == 0:
                print(f'[{state.iterations}]:\t {self.formatValue(self.lastValue)}')

    def formatValue(self, value):
        return str(value)

    @abstractmethod
    def calculate(self, state: LatticeState):
        pass


class OrderParametersCalculator(Calculator):
    def __init__(self, orderParametersHistory: OrderParametersHistory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orderParametersHistory = orderParametersHistory

    def calculate(self, state: LatticeState):
        op = calculateOrderParameters(state)
        self.orderParametersHistory.orderParameters = np.append(self.orderParametersHistory.orderParameters, op)
        return op

    def formatValue(self, value):
        s = ','.join([f'{name}={value[name][0]:.5f}' for name in gatheredOrderParameters.fields.keys()])
        return 'averg: '+s


class FluctuationsCalculator(Calculator):
    def __init__(self, orderParametersHistory: OrderParametersHistory, *args, window=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.orderParametersHistory = orderParametersHistory

    def calculate(self, state):
        fluctuations = np.zeros(1, dtype=gatheredOrderParameters)
        for name in gatheredOrderParameters.fields.keys():
            fluct = state.lattice.particles.size * fluctuation(self.orderParametersHistory.orderParameters[name][-100:])
            fluctuations[name] = fluct

        self.orderParametersHistory.fluctuations = np.append(self.orderParametersHistory.fluctuations, fluctuations)
        return fluctuations

    def formatValue(self, value):
        s = ','.join([f'{name}={value[name][0]:.5f}' for name in gatheredOrderParameters.fields.keys()])
        return 'fluct: '+s


class RandomWiggleRateAdjustor(Calculator):
    def __init__(self, scale, *args, resetValue=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.resetValue = resetValue

    def calculate(self, state):
        if self.resetValue is not None:
            state.wiggleRate = np.random.normal(self.resetValue, scale=self.scale)
        else:
            state.wiggleRate = np.random.normal(state.wiggleRate, scale=self.scale)
        return state.wiggleRate


class DerivativeWiggleRateAdjustor(Calculator):
    def __init__(self, howMany, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.howMany = howMany

    def calculate(self, state):
        mE = np.array([m.mean() for m in np.split(state.latticeAverages['energy'][-self.howMany:], 4)])
        mR = np.array([m.mean() for m in np.split(state.wiggleRateValues[-self.howMany:], 4)])
        mR[np.where(mR == 0)] = np.random.normal(scale=0.001)

        de = np.diff(mE, 1)
        dr = np.diff(mR, 1)
        efirst = de/dr
        etrend = (efirst[-1]-efirst[-2])
        if etrend < 0:
            state.wiggleRate *= 1.1
        else:
            state.wiggleRate *= 0.9
        return state.wiggleRate
