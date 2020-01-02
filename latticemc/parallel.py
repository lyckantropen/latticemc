from .definitions import LatticeState, OrderParametersHistory, DefiningParameters
from .updaters import CallbackUpdater, FluctuationsCalculator, OrderParametersCalculator, Updater
from .failsafe import failsafeSaveSimulation
from . import simulationNumba
import multiprocessing as mp
import threading
import numpy as np
from enum import Enum
from collections import namedtuple
from typing import Dict, List


class MessageType(Enum):
    OrderParameters = 1
    Fluctuations = 2
    State = 3
    ParallelTemperingSignUp = 4


ParallelTemperingParameters = namedtuple('ParallelTemperingParameters', ['parameters', 'energy', 'pipe'])


class SimulationProcess(mp.Process):
    def __init__(self,
                 queue: mp.Queue,
                 initialState: LatticeState,
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
        self.state = initialState
        self.cycles = cycles
        self.reportOrderParametersEvery = reportOrderParametersEvery
        self.reportStateEvery = reportStateEvery
        self.perStateUpdaters = perStateUpdaters
        self.fluctuationsHowOften = fluctuationsHowOften
        self.fluctuationsWindow = fluctuationsWindow
        self.parallelTemperingInterval = parallelTemperingInterval

    def run(self):
        localHistory = {self.state.parameters: OrderParametersHistory()}

        orderParametersBroadcaster = CallbackUpdater(
            callback=lambda _: self.queue.put((MessageType.OrderParameters, (self.state.parameters, localHistory[self.state.parameters].orderParameters[-self.reportOrderParametersEvery:]))),
            howOften=self.reportOrderParametersEvery,
            sinceWhen=self.reportOrderParametersEvery
        )
        fluctuationsBroadcaster = CallbackUpdater(
            callback=lambda _: self.queue.put((MessageType.Fluctuations, (self.state.parameters, localHistory[self.state.parameters].fluctuations[-self.fluctuationsHowOften:]))),
            howOften=self.fluctuationsHowOften,
            sinceWhen=self.fluctuationsWindow
        )
        stateBroadcaster = CallbackUpdater(
            callback=lambda _: self.queue.put((MessageType.State, self.state)),
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
                callback=lambda _: self._parallelTempering(localHistory),
                howOften=self.parallelTemperingInterval,
                sinceWhen=self.parallelTemperingInterval
            )
            perStateUpdaters.append(parallelTemperingUpdater)

        try:
            for it in range(self.cycles):
                simulationNumba.doLatticeStateUpdate(self.state)
                for u in perStateUpdaters:
                    u.perform(self.state)

        except Exception as e:
            failsafeSaveSimulation(e, self.state, localHistory)

        self.queue.put((MessageType.State, self.state))

    def _parallelTempering(self, localHistory: Dict[DefiningParameters, OrderParametersHistory]):
        """
        Post a message to the queue that this configuration is ready
        for parallel tempering. Open a pipe and wait for a new set
        of parameters, then change them if they are different.
        """
        energy = localHistory[self.state.parameters].orderParameters['energy'][-1] * self.state.lattice.particles.size
        our, theirs = mp.Pipe()
        self.queue.put((MessageType.ParallelTemperingSignUp, ParallelTemperingParameters(parameters=self.state.parameters, energy=energy, pipe=theirs)))

        # wait for decision in governing thread
        parameters = our.recv()
        if parameters != self.state.parameters:
            # parameter change
            hist = localHistory.pop(self.state.parameters)
            localHistory[parameters] = hist
            self.state.parameters = parameters


class SimulationRunner(threading.Thread):
    def __init__(self,
                 initialStates: List[LatticeState],
                 orderParametersHistory: Dict[DefiningParameters, OrderParametersHistory],
                 *args,
                 **kwargs):
        threading.Thread.__init__(self)

        self.states = initialStates
        self.orderParametersHistory = orderParametersHistory
        self.args = args
        self.kwargs = kwargs
        self.simulations = []
        self._starting = True

    def run(self):
        q = mp.Queue()

        self.simulations = []
        for state in self.states:
            sim = SimulationProcess(q, state, *self.args, **self.kwargs)
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
                    # update the state
                    [state for state in self.states if state.parameters == msg.parameters][0] = msg
                if messageType == MessageType.ParallelTemperingSignUp:
                    # add this state to the waiting list for parallel tempering
                    ptReady[msg.parameters.temperature] = msg

            # process waiting list for parallel tempering in random order
            if ptReady:
                for _ in range(len(ptReady)):
                    self._doParallelTempering(ptReady)

        [sim.join() for sim in self.simulations]

    def _doParallelTempering(self, ptReady: Dict[float, ParallelTemperingParameters]):
        # only consider adjacent temperatures
        temperatures = sorted([state.parameters.temperature for state in self.states])
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
        t1, e1, pipe1 = p1.parameters.temperature, p1.energy, p1.pipe
        t2, e2, pipe2 = p2.parameters.temperature, p2.energy, p2.pipe
        dB = 1 / t1 - 1 / t2
        dE = e1 - e2
        if dB * dE > 0 or np.random.random() < np.exp(dB * dE):
            # sending new parameters down the pipe will unblock waiting processes
            pipe1.send(p2.parameters)
            pipe2.send(p1.parameters)
            return True
        else:
            # sending old parameters down the pipe will unblock waiting processes
            pipe1.send(p1.parameters)
            pipe2.send(p2.parameters)
            return False

    def stop(self):
        [sim.terminate() for sim in self.simulations]

    def alive(self):
        return self._starting or [sim for sim in self.simulations if sim.is_alive()]
