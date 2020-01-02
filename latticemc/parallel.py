from .definitions import Lattice, LatticeState, OrderParametersHistory, DefiningParameters
from .updaters import CallbackUpdater, FluctuationsCalculator, OrderParametersCalculator, Updater
from .randomQuaternion import randomQuaternion
from .latticeTools import initializePartiallyOrdered
from .failsafe import failsafeSaveSimulation
from . import simulationNumba
import multiprocessing as mp
import threading
import numpy as np
from enum import Enum
from collections import namedtuple
from typing import Dict, Tuple, List


class MessageType(Enum):
    OrderParameters = 1
    Fluctuations = 2
    State = 3
    ParallelTemperingSignUp = 4


ParallelTemperingParameters = namedtuple('ParallelTemperingParameters', ['params', 'energy', 'pipe'])


class SimulationProcess(mp.Process):
    def __init__(self,
                 queue: mp.Queue,
                 params: DefiningParameters,
                 latticeSize: Tuple[int, int, int],
                 cycles: int,
                 reportOrderParametersEvery: int = 1000,
                 reportStateEvery: int = 1000,
                 fluctuationsHowOften: int = 50,
                 fluctuationsWindow: int = 100,
                 perStateUpdaters: List[Updater] = [],
                 parallelTemperingInterval: int = None
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
        self.parallelTemperingInterval = parallelTemperingInterval

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
            callback=lambda state: self.queue.put((MessageType.Fluctuations, (self.params, localHistory[self.params].fluctuations[-self.fluctuationsHowOften:]))),
            howOften=self.fluctuationsHowOften,
            sinceWhen=self.fluctuationsWindow
        )
        stateBroadcaster = CallbackUpdater(
            callback=lambda state: self.queue.put((MessageType.State, state)),
            howOften=self.reportStateEvery,
            sinceWhen=self.reportStateEvery
        )

        orderParametersCalculator = OrderParametersCalculator(localHistory, howOften=1, sinceWhen=0)
        fluctuationsCalculator = FluctuationsCalculator(localHistory, window=self.fluctuationsWindow, howOften=self.fluctuationsHowOften, sinceWhen=self.fluctuationsWindow)
        perStateUpdaters = [
            orderParametersCalculator,
            fluctuationsCalculator,

            *self.perStateUpdaters,

            orderParametersBroadcaster,
            fluctuationsBroadcaster,
            stateBroadcaster
        ]

        if self.parallelTemperingInterval is not None:
            parallelTemperingUpdater = CallbackUpdater(
                callback=lambda state: self._parallelTempering(localHistory, state),
                howOften=self.parallelTemperingInterval,
                sinceWhen=self.parallelTemperingInterval
            )
            perStateUpdaters.append(parallelTemperingUpdater)

        try:
            for it in range(self.cycles):
                simulationNumba.doLatticeStateUpdate(state)
                for u in perStateUpdaters:
                    u.perform(state)

        except Exception as e:
            failsafeSaveSimulation(e, state, localHistory)

        self.queue.put((MessageType.State, state))

    def _parallelTempering(self, localHistory: Dict[DefiningParameters, OrderParametersHistory], state: LatticeState):
        """
        Post a message to the queue that this configuration is ready
        for parallel tempering. Open a pipe and wait for a new set
        of parameters, then change them if they are different.
        """
        energy = localHistory[self.params].orderParameters['energy'][-1] * np.multiply.reduce(self.latticeSize)
        our, theirs = mp.Pipe()
        self.queue.put((MessageType.ParallelTemperingSignUp, ParallelTemperingParameters(params=self.params, energy=energy, pipe=theirs)))

        # wait for decision in governing thread
        params = our.recv()
        if params != self.params:
            # parameter change
            hist = localHistory.pop(self.params)
            localHistory[params] = hist
            self.params = params
            state.parameters = params


class SimulationRunner(threading.Thread):
    def __init__(self,
                 parameters: DefiningParameters,
                 orderParametersHistory: Dict[DefiningParameters, OrderParametersHistory],
                 *args,
                 **kwargs):
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

        ptReady = {}
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
                    pass  # TODO
                if messageType == MessageType.ParallelTemperingSignUp:
                    # add this state to the waiting list for parallel tempering
                    ptReady[msg.params.temperature] = msg

            # process waiting list for parallel tempering in random order
            if ptReady:
                for _ in range(len(ptReady)):
                    self._doParallelTempering(ptReady)

        [sim.join() for sim in self.simulations]

    def _doParallelTempering(self, ptReady: Dict[float, ParallelTemperingParameters]):
        # only consider adjacent temperatures
        temperatures = sorted([p.temperature for p in self.parameters])
        i = np.random.randint(len(temperatures) - 1)
        j = i + 1
        ti = temperatures[i]
        tj = temperatures[j]

        # check if adjacent temperatures exist in waiting list
        pi = ptReady[ti] if ti in ptReady else None
        pj = ptReady[tj] if tj in ptReady else None
        if pi is not None and pj is not None:
            self._doParallelTemperingDecision(pi, pj)
            # remove the pair from the pool
            ptReady.pop(ti)
            ptReady.pop(tj)

    def _doParallelTemperingDecision(self, p1: ParallelTemperingParameters, p2: ParallelTemperingParameters):
        t1, e1, pipe1 = p1.params.temperature, p1.energy, p1.pipe
        t2, e2, pipe2 = p2.params.temperature, p2.energy, p2.pipe
        dB = 1 / t1 - 1 / t2
        dE = e1 - e2
        if dB * dE > 0 or np.random.random() < np.exp(dB * dE):
            # sending new parameters down the pipe will unblock waiting processes
            pipe1.send(p2.params)
            pipe2.send(p1.params)
            return True
        else:
            # sending old parameters down the pipe will unblock waiting processes
            pipe1.send(p1.params)
            pipe2.send(p2.params)
            return False

    def stop(self):
        [sim.terminate() for sim in self.simulations]

    def alive(self):
        return self._starting or [sim for sim in self.simulations if sim.is_alive()]
