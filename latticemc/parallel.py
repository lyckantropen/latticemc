from .definitions import Lattice, LatticeState, OrderParametersHistory, DefiningParameters
from .updaters import CallbackUpdater, FluctuationsCalculator, OrderParametersCalculator
from .randomQuaternion import randomQuaternion
from .latticeTools import initializePartiallyOrdered
from .failsafe import failsafeSaveSimulation
from . import simulationNumba
import multiprocessing as mp
import threading
import numpy as np
from enum import Enum


class MessageType(Enum):
    OrderParameters = 1
    Fluctuations = 2
    State = 3


class SimulationProcess(mp.Process):
    def __init__(self,
                 queue: mp.Queue,
                 params: DefiningParameters,
                 latticeSize: tuple,
                 cycles: int,
                 reportOrderParametersEvery: int = 1000,
                 reportStateEvery: int = 1000,
                 fluctuationsHowOften: int = 50,
                 fluctuationsWindow: int = 100,
                 perStateUpdaters: list = []
                 ):
        super().__init__()
        self.queue = queue
        self.params = params
        self.cycles = cycles
        self.latticeSize = latticeSize
        self.reportOrderParametersEvery = reportOrderParametersEvery
        self.reportStateEvery = reportStateEvery
        self.perStateUpdaters = perStateUpdaters
        self.fluctuationsHowOften = fluctuationsHowOften
        self.fluctuationsWindow = fluctuationsWindow

    def run(self):
        state = LatticeState(parameters=self.params, lattice=Lattice(*self.latticeSize))
        initializePartiallyOrdered(state.lattice, x=randomQuaternion(1))
        localHistory = {self.params: OrderParametersHistory()}

        orderParametersBroadcaster = CallbackUpdater(
            callback=lambda state: self.queue.put((MessageType.OrderParameters, (self.params, localHistory[self.params].orderParameters[-self.reportOrderParametersEvery:]))),
            howOften=self.reportOrderParametersEvery,
            sinceWhen=self.reportOrderParametersEvery
        )
        fluctuationsBroadcaster = CallbackUpdater(
            callback=lambda state: self.queue.put((MessageType.Fluctuations, (self.params, localHistory[self.params].fluctuations[-50:]))),
            howOften=self.fluctuationsHowOften,
            sinceWhen=self.fluctuationsWindow
        )
        stateBroadcaster = CallbackUpdater(
            callback=lambda state: self.queue.put((MessageType.State, state)),
            howOften=self.reportStateEvery,
            sinceWhen=self.reportStateEvery
        )

        orderParametersCalculator = OrderParametersCalculator(localHistory, howOften=1, sinceWhen=1)
        fluctuationsCalculator = FluctuationsCalculator(localHistory, window=self.fluctuationsWindow, howOften=self.fluctuationsHowOften, sinceWhen=self.fluctuationsWindow)
        perStateUpdaters = [
            orderParametersCalculator,
            fluctuationsCalculator,

            *self.perStateUpdaters,

            orderParametersBroadcaster,
            fluctuationsBroadcaster,
            stateBroadcaster
        ]
        try:
            for it in range(self.cycles):
                simulationNumba.doLatticeStateUpdate(state)
                for u in perStateUpdaters:
                    u.perform(state)
        except Exception as e:
            failsafeSaveSimulation(e, state, localHistory)

        self.queue.put((MessageType.State, state))


class SimulationRunner(threading.Thread):
    def __init__(self, parameters, orderParametersHistory, *args, **kwargs):
        threading.Thread.__init__(self)

        self.parameters = parameters
        self.orderParametersHistory = orderParametersHistory
        self.args = args
        self.kwargs = kwargs
        self.simulations = []
        self._starting = True

    def run(self):
        q = mp.Queue()

        self.simulations = []
        for p in self.parameters:
            sim = SimulationProcess(q, p, *self.args, **self.kwargs)
            sim.start()
            self.simulations.append(sim)

        self._starting = False
        while self.alive():
            while not q.empty():
                messageType, msg = q.get()
                if messageType == MessageType.OrderParameters:
                    p, op = msg
                    self.orderParametersHistory[p].orderParameters = np.append(self.orderParametersHistory[p].orderParameters, op)
                if messageType == MessageType.Fluctuations:
                    p, fl = msg
                    self.orderParametersHistory[p].fluctuations = np.append(self.orderParametersHistory[p].fluctuations, fl)
                if messageType == MessageType.State:
                    pass
        [sim.join() for sim in self.simulations]

    def stop(self):
        [sim.terminate() for sim in self.simulations]

    def alive(self):
        return self._starting or [sim for sim in self.simulations if sim.is_alive()]
