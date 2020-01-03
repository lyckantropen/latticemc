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

        self._relevantHistoryLength = 0
        self.localHistory = OrderParametersHistory()

    def run(self):
        orderParametersBroadcaster = CallbackUpdater(
            callback=lambda _: self._broadcastOrderParameters(),
            howOften=self.reportOrderParametersEvery,
            sinceWhen=self.reportOrderParametersEvery
        )
        fluctuationsBroadcaster = CallbackUpdater(
            callback=lambda _: self._broadcastFluctuations(),
            howOften=self.fluctuationsHowOften,
            sinceWhen=self.fluctuationsWindow
        )
        stateBroadcaster = CallbackUpdater(
            callback=lambda _: self._broadcastState(),
            howOften=self.reportStateEvery,
            sinceWhen=self.reportStateEvery
        )

        orderParametersCalculator = OrderParametersCalculator(self.localHistory, howOften=1, sinceWhen=0)
        fluctuationsCalculator = FluctuationsCalculator(self.localHistory, window=self.fluctuationsWindow, howOften=self.fluctuationsHowOften, sinceWhen=self.fluctuationsWindow)
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
                callback=lambda _: self._parallelTempering(),
                howOften=self.parallelTemperingInterval,
                sinceWhen=self.parallelTemperingInterval
            )
            perStateUpdaters.append(parallelTemperingUpdater)

        try:
            for it in range(self.cycles):
                simulationNumba.doLatticeStateUpdate(self.state)
                self._relevantHistoryLength += 1
                for u in perStateUpdaters:
                    u.perform(self.state)

        except Exception as e:
            failsafeSaveSimulation(e, self.state, self.localHistory)

        self.queue.put((MessageType.State, self.state))

    def _broadcastOrderParameters(self):
        self.queue.put((MessageType.OrderParameters,
                        (self.state.parameters,
                         self.localHistory.orderParameters[-min(self._relevantHistoryLength, self.reportOrderParametersEvery):])))

    def _broadcastFluctuations(self):
        self.queue.put((MessageType.Fluctuations,
                        (self.state.parameters,
                         self.localHistory.fluctuations[-min(self._relevantHistoryLength, self.fluctuationsHowOften):])))

    def _broadcastState(self):
        self.queue.put((MessageType.State, self.state))

    def _parallelTempering(self):
        """
        Post a message to the queue that this configuration is ready
        for parallel tempering. Open a pipe and wait for a new set
        of parameters, then change them if they are different.
        """
        energy = self.localHistory.orderParameters['energy'][-1] * self.state.lattice.particles.size
        our, theirs = mp.Pipe()
        self.queue.put((MessageType.ParallelTemperingSignUp, ParallelTemperingParameters(parameters=self.state.parameters, energy=energy, pipe=theirs)))

        # wait for decision in governing thread
        if not our.poll(30):
            print(f'{self.state.parameters}: No parallel tempering data to exchange')
        else:
            parameters = our.recv()
            if parameters != self.state.parameters:
                # broadcast what we can
                self._broadcastOrderParameters()
                self._broadcastFluctuations()
                self._broadcastState()

                # parameter change
                self.state.parameters = parameters

                # reset the number of results that can be safely broadcasted as coming from this configuration
                self._relevantHistoryLength = 0


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

        # for parallel tempering
        self._temperatures = [state.parameters.temperature for state in self.states]

    def run(self):
        q = mp.Queue()

        self.simulations = []
        for i, state in enumerate(self.states):
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
                    state = [state for state in self.states if state.parameters == msg.parameters][0]
                    state.iterations = msg.iterations
                    state.lattice = msg.lattice
                    state.latticeAverages = msg.latticeAverages
                    state.wiggleRate = msg.wiggleRate
                if messageType == MessageType.ParallelTemperingSignUp:
                    # add this state to the waiting list for parallel tempering
                    temperatureIndex = self._temperatures.index(msg.parameters.temperature)
                    ptReady[temperatureIndex] = msg

            # process waiting list for parallel tempering in random order
            if ptReady:
                for _ in range(len(ptReady)):
                    self._doParallelTempering(ptReady)

        [sim.join() for sim in self.simulations]

    def _doParallelTempering(self, ptReady: Dict[float, ParallelTemperingParameters]):
        # only consider adjacent temperatures
        i = np.random.randint(len(self._temperatures) - 1)
        j = i + 1

        # check if adjacent temperatures exist in waiting list
        pi = ptReady[i] if i in ptReady else None
        pj = ptReady[j] if j in ptReady else None
        if pi is not None and pj is not None:
            self._doParallelTemperingDecision(pi, pj)
            # remove the pair from the pool
            ptReady.pop(i)
            ptReady.pop(j)

    def _doParallelTemperingDecision(self, p1: ParallelTemperingParameters, p2: ParallelTemperingParameters):
        t1, e1, pipe1 = float(p1.parameters.temperature), p1.energy, p1.pipe
        t2, e2, pipe2 = float(p2.parameters.temperature), p2.energy, p2.pipe
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
