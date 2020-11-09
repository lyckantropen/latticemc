import logging
import multiprocessing as mp
import threading
from collections import namedtuple
from ctypes import c_double
from enum import Enum
from typing import Dict, List

import numpy as np

from . import simulationNumba
from .definitions import (DefiningParameters, LatticeState,
                          OrderParametersHistory)
from .failsafe import failsafeSaveSimulation
from .updaters import (CallbackUpdater, FluctuationsCalculator,
                       OrderParametersCalculator, Updater)

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """
    Messages that SimulationProcess can send to Simulation Runner.
    All messages have the format of a tuple:
    (MessageType, self.index, payload)
    """
    OrderParameters = 1
    Fluctuations = 2
    State = 3
    ParallelTemperingSignUp = 4
    Error = 5
    Finished = 6


ParallelTemperingParameters = namedtuple('ParallelTemperingParameters', ['parameters', 'energy', 'pipe'])


class SimulationProcess(mp.Process):
    """
    A process representing one simulation over a lattice of particles.
    When parallel tempering is enabled, does not necessarily mean a
    particular configuration of parameters. Otherwise the parameters
    are constant.
    """

    def __init__(self,
                 index: int,
                 queue: mp.Queue,
                 initialState: LatticeState,
                 cycles: int,
                 reportOrderParametersEvery: int = 1000,
                 reportFluctuationsEvery: int = 1000,
                 reportStateEvery: int = 1000,
                 fluctuationsWindow: int = 1000,
                 perStateUpdaters: List[Updater] = [],
                 parallelTemperingInterval: int = None
                 ):
        super().__init__()
        self.index = index
        self.queue = queue
        self.state = initialState
        self.cycles = cycles
        self.reportOrderParametersEvery = reportOrderParametersEvery
        self.reportFluctuationsEvery = reportFluctuationsEvery
        self.reportStateEvery = reportStateEvery
        self.perStateUpdaters = perStateUpdaters
        self.fluctuationsWindow = fluctuationsWindow
        self.parallelTemperingInterval = parallelTemperingInterval

        self.running = mp.Value('i', 1)
        self.temperature = mp.Value(c_double, float(self.state.parameters.temperature))

        # how many data points truly belong to the present configuration
        # is important when parallel tempering is enabled
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
            howOften=self.reportFluctuationsEvery,
            sinceWhen=self.fluctuationsWindow
        )
        stateBroadcaster = CallbackUpdater(
            callback=lambda _: self._broadcastState(),
            howOften=self.reportStateEvery,
            sinceWhen=self.reportStateEvery
        )

        orderParametersCalculator = OrderParametersCalculator(self.localHistory, howOften=1, sinceWhen=0)
        fluctuationsCalculator = FluctuationsCalculator(self.localHistory, window=self.fluctuationsWindow, howOften=1, sinceWhen=self.fluctuationsWindow)
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
            self.queue.put((MessageType.Error, self.index, (self.state.parameters, e)))
            failsafeSaveSimulation(e, self.state, self.localHistory)
            self.running.value = 0

        self.queue.put((MessageType.State, self.index, self.state))
        self.queue.put((MessageType.Finished, self.index, self.state.parameters))
        self.running.value = 0

    def _broadcastOrderParameters(self):
        """
        Publish at most self._relevantHistoryLength order
        parameters from history to the governing thread.
        """
        self.queue.put((MessageType.OrderParameters, self.index,
                        (self.state.parameters,
                         self.localHistory.orderParameters[-min(self._relevantHistoryLength, self.reportOrderParametersEvery):])))

    def _broadcastFluctuations(self):
        """
        Publish at most self._relevantHistoryLength fluctuation
        values from history to the governing thread.
        """
        self.queue.put((MessageType.Fluctuations, self.index,
                        (self.state.parameters,
                         self.localHistory.fluctuations[-min(self._relevantHistoryLength, self.reportFluctuationsEvery):])))

    def _broadcastState(self):
        """
        Publish the current Lattice State to the
        governing thread.
        """
        self.queue.put((MessageType.State, self.index, self.state))

    def _parallelTempering(self):
        """
        Post a message to the queue that this configuration is ready
        for parallel tempering. Open a pipe and wait for a new set
        of parameters, then change them if they are different.

        Upon change, publish the relevant order parameters history.
        """
        energy = self.localHistory.orderParameters['energy'][-1] * self.state.lattice.particles.size
        our, theirs = mp.Pipe()
        self.queue.put((MessageType.ParallelTemperingSignUp, self.index, ParallelTemperingParameters(parameters=self.state.parameters, energy=energy, pipe=theirs)))

        # wait for decision in governing thread
        if not our.poll(30):
            logger.warning(f'SimulationProcess[{self.index}, {self.state.parameters}]: No parallel tempering data to exchange')
        parameters = our.recv()

        logger.debug(f'SimulationProcess[{self.index}, {self.state.parameters}]: Received parameters for exchange: {parameters}')

        if parameters != self.state.parameters:
            # broadcast what we can
            self._broadcastOrderParameters()
            self._broadcastFluctuations()
            self._broadcastState()

            # parameter change
            with self.temperature.get_lock():
                self.state.parameters = parameters
                self.temperature.value = float(parameters.temperature)

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
        self.simulations: List[SimulationProcess] = []

        # set to False when all processes have started running
        self._starting = True

        # for parallel tempering
        self._temperatures = [state.parameters.temperature for state in self.states]

    def run(self):
        q = mp.Queue()

        self.simulations = []
        for i, state in enumerate(self.states):
            sim = SimulationProcess(i, q, state, *self.args, **self.kwargs)
            sim.start()
            self.simulations.append(sim)

        self._starting = False

        ptReady: List[ParallelTemperingParameters] = []
        while self.alive():
            while not q.empty():
                messageType, index, msg = q.get()
                logger.debug(f'SimulationRunner: Received {messageType}, index={index}')

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
                    ptParameters = msg
                    ptReady.append(ptParameters)
                if messageType == MessageType.Error:
                    parameters, exception = msg
                    logger.error(f'SimulationProcess[{index},{parameters}]: Failed with exception "{exception}"')
                if messageType == MessageType.Finished:
                    parameters = msg
                    logger.info(f'SimulationProcess[{index},{parameters}]: Finished succesfully')

            self._doParallelTempering(ptReady)

        [sim.join() for sim in self.simulations]

    def _adjacentTemperature(self, pi: ParallelTemperingParameters):
        """
        Find the value of adjacent temperature within the
        values that are present in the simulations.
        """
        tempIndex = self._temperatures.index(pi.parameters.temperature)
        if tempIndex + 1 == len(self._temperatures):
            return self._temperatures[tempIndex - 1]
        else:
            return self._temperatures[tempIndex + 1]

    def _simulationRunningThisTemperature(self, temperature: float):
        """
        Return the running simulation that currently has
        'temperature' set as the temperature it is running at.
        If no such simulation can be found, return None.
        """
        simsForTemp = [sim for sim in self.simulations if np.isclose(sim.temperature.value, temperature) and sim.running.value]
        if simsForTemp:
            return simsForTemp[0]
        else:
            return None

    def _doParallelTempering(self, ptReady: List[ParallelTemperingParameters]):
        """
        Manage random selection of temperatures and exchanging
        parameters between configurations using parameters and pipes provided
        in 'ptReady'.
        """
        # process waiting list for parallel tempering in random order
        import random
        ptParam = None
        it = len(ptReady)
        while it > 0 and ptReady:
            ptParam = random.choice(ptReady)
            adjTemp = self._adjacentTemperature(ptParam)
            if self._simulationRunningThisTemperature(float(adjTemp)) is None:
                # unblock
                ptParam.pipe.send(ptParam.parameters)
                ptReady.remove(ptParam)
                logger.debug(f'SimulationRunner: Freed up {ptParam} from PT waiting list')
            else:
                try:
                    adjPtParam = [p for p in ptReady if p.parameters.temperature == adjTemp][0]
                    if self._parallelTemperingDecision(ptParam, adjPtParam):
                        # sending new parameters down the pipe will unblock waiting processes
                        ptParam.pipe.send(adjPtParam.parameters)
                        adjPtParam.pipe.send(ptParam.parameters)
                        logger.debug(f'SimulationRunner: Exchanged {ptParam.parameters.temperature} and {adjPtParam.parameters.temperature}')
                    else:
                        # sending old parameters down the pipe will unblock waiting processes
                        ptParam.pipe.send(ptParam.parameters)
                        adjPtParam.pipe.send(adjPtParam.parameters)
                        logger.debug(f'SimulationRunner: Did not exchange {ptParam.parameters.temperature} and {adjPtParam.parameters.temperature}')

                    ptReady.remove(ptParam)
                    ptReady.remove(adjPtParam)
                except IndexError:
                    pass
            it -= 1

    @staticmethod
    def _parallelTemperingDecision(p1: ParallelTemperingParameters, p2: ParallelTemperingParameters) -> bool:
        t1, e1, _ = float(p1.parameters.temperature), p1.energy, p1.pipe
        t2, e2, _ = float(p2.parameters.temperature), p2.energy, p2.pipe
        dB = 1 / t1 - 1 / t2
        dE = e1 - e2
        return dB * dE > 0 or np.random.random() < np.exp(dB * dE)

    def stop(self):
        [sim.terminate() for sim in self.simulations]

    def alive(self):
        return self._starting or [sim for sim in self.simulations if sim.is_alive() or sim.running.value]
