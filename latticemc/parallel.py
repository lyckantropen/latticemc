import logging
import multiprocessing as mp
import threading
from collections import namedtuple
from ctypes import c_double
from enum import Enum
from typing import Dict, List

import numpy as np

from . import simulation_numba
from .definitions import DefiningParameters, LatticeState, OrderParametersHistory
from .failsafe import failsafe_save_simulation
from .updaters import CallbackUpdater, FluctuationsCalculator, OrderParametersCalculator, Updater

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """
    The type of message that `SimulationProcess` can send to `SimulationRunner`.

    All messages have the format of a tuple: `(MessageType, self.index, payload)`
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
    Rrocess representing one simulation over a lattice of particles.

    When parallel tempering is enabled, does not necessarily mean a
    particular configuration of parameters. Otherwise the parameters
    are constant.
    """

    def __init__(self,
                 index: int,
                 queue: mp.Queue,
                 initial_state: LatticeState,
                 cycles: int,
                 report_order_parameters_every: int = 1000,
                 report_fluctuations_every: int = 1000,
                 report_state_every: int = 1000,
                 fluctuations_window: int = 1000,
                 per_state_updaters: List[Updater] = [],
                 parallel_tempering_interval: int = None
                 ):
        super().__init__()
        self.index = index
        self.queue = queue
        self.state = initial_state
        self.cycles = cycles
        self.report_order_parameters_every = report_order_parameters_every
        self.report_fluctuations_every = report_fluctuations_every
        self.report_state_every = report_state_every
        self.per_state_updaters = per_state_updaters
        self.fluctuations_window = fluctuations_window
        self.parallel_tempering_interval = parallel_tempering_interval

        self.running = mp.Value('i', 1)
        self.temperature = mp.Value(c_double, float(self.state.parameters.temperature))

        # how many data points truly belong to the present configuration
        # is important when parallel tempering is enabled
        self._relevant_history_length = 0
        self.local_history = OrderParametersHistory()

    def run(self):
        order_parameters_broadcaster = CallbackUpdater(
            callback=lambda _: self._broadcast_order_parameters(),
            how_often=self.report_order_parameters_every,
            since_when=self.report_order_parameters_every
        )
        fluctuations_broadcaster = CallbackUpdater(
            callback=lambda _: self._broadcast_fluctuations(),
            how_often=self.report_fluctuations_every,
            since_when=self.fluctuations_window
        )
        state_broadcaster = CallbackUpdater(
            callback=lambda _: self._broadcast_state(),
            how_often=self.report_state_every,
            since_when=self.report_state_every
        )

        order_parameters_calculator = OrderParametersCalculator(self.local_history, how_often=1, since_when=0)
        fluctuations_calculator = FluctuationsCalculator(self.local_history, window=self.fluctuations_window, how_often=1, since_when=self.fluctuations_window)
        per_state_updaters = [
            order_parameters_calculator,
            fluctuations_calculator,

            *self.per_state_updaters,

            order_parameters_broadcaster,
            fluctuations_broadcaster,
            state_broadcaster
        ]

        if self.parallel_tempering_interval is not None:
            parallel_tempering_updater = CallbackUpdater(
                callback=lambda _: self._parallel_tempering(),
                how_often=self.parallel_tempering_interval,
                since_when=self.parallel_tempering_interval
            )
            per_state_updaters.append(parallel_tempering_updater)

        try:
            for _ in range(self.cycles):
                simulation_numba.do_lattice_state_update(self.state)
                self._relevant_history_length += 1
                for u in per_state_updaters:
                    u.perform(self.state)

        except Exception as e:
            self.queue.put((MessageType.Error, self.index, (self.state.parameters, e)))
            failsafe_save_simulation(e, self.state, self.local_history)
            self.running.value = 0

        self.queue.put((MessageType.State, self.index, self.state))
        self.queue.put((MessageType.Finished, self.index, self.state.parameters))
        self.running.value = 0

    def _broadcast_order_parameters(self):
        """Publish at most `self._relevant_history_length` order parameters from history to the governing thread."""
        self.queue.put((MessageType.OrderParameters, self.index,
                        (self.state.parameters,
                         self.local_history.order_parameters[-min(self._relevant_history_length, self.report_order_parameters_every):])))

    def _broadcast_fluctuations(self):
        """Publish at most `self._relevant_history_length` fluctuation values from history to the governing thread."""
        self.queue.put((MessageType.Fluctuations, self.index,
                        (self.state.parameters,
                         self.local_history.fluctuations[-min(self._relevant_history_length, self.report_fluctuations_every):])))

    def _broadcast_state(self):
        """Publish the current Lattice State to the governing thread."""
        self.queue.put((MessageType.State, self.index, self.state))

    def _parallel_tempering(self):
        """
        Report readiness for parallel tempering update.

        Post a message to the queue that this configuration is ready
        for parallel tempering. Open a pipe and wait for a new set
        of parameters, then change them if they are different.
        Upon change, publish the relevant order parameters history.
        """
        energy = self.local_history.order_parameters['energy'][-1] * self.state.lattice.particles.size
        our, theirs = mp.Pipe()
        self.queue.put((MessageType.ParallelTemperingSignUp, self.index, ParallelTemperingParameters(
            parameters=self.state.parameters, energy=energy, pipe=theirs)))

        # wait for decision in governing thread
        if not our.poll(30):
            logger.warning(f'SimulationProcess[{self.index}, {self.state.parameters}]: No parallel tempering data to exchange')
        parameters = our.recv()

        logger.debug(f'SimulationProcess[{self.index}, {self.state.parameters}]: Received parameters for exchange: {parameters}')

        if parameters != self.state.parameters:
            # broadcast what we can
            self._broadcast_order_parameters()
            self._broadcast_fluctuations()
            self._broadcast_state()

            # parameter change
            with self.temperature.get_lock():
                self.state.parameters = parameters
                self.temperature.value = float(parameters.temperature)

            # reset the number of results that can be safely broadcasted as coming from this configuration
            self._relevant_history_length = 0


class SimulationRunner(threading.Thread):
    def __init__(self,
                 initial_states: List[LatticeState],
                 order_parameters_history: Dict[DefiningParameters, OrderParametersHistory],
                 *args,
                 **kwargs):
        threading.Thread.__init__(self)

        self.states = initial_states
        self.order_parameters_history = order_parameters_history
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

        pt_ready: List[ParallelTemperingParameters] = []
        while self.alive():
            while not q.empty():
                message_type, index, msg = q.get()
                logger.debug(f'SimulationRunner: Received {message_type}, index={index}')

                if message_type == MessageType.OrderParameters:
                    p, op = msg
                    self.order_parameters_history[p].order_parameters = np.append(self.order_parameters_history[p].order_parameters, op)
                if message_type == MessageType.Fluctuations:
                    p, fl = msg
                    self.order_parameters_history[p].fluctuations = np.append(self.order_parameters_history[p].fluctuations, fl)
                if message_type == MessageType.State:
                    # update the state
                    state = [state for state in self.states if state.parameters == msg.parameters][0]
                    state.iterations = msg.iterations
                    state.lattice = msg.lattice
                    state.lattice_averages = msg.lattice_averages
                    state.wiggle_rate = msg.wiggle_rate
                if message_type == MessageType.ParallelTemperingSignUp:
                    pt_parameters = msg
                    pt_ready.append(pt_parameters)
                if message_type == MessageType.Error:
                    parameters, exception = msg
                    logger.error(f'SimulationProcess[{index},{parameters}]: Failed with exception "{exception}"')
                if message_type == MessageType.Finished:
                    parameters = msg
                    logger.info(f'SimulationProcess[{index},{parameters}]: Finished succesfully')

            self._do_parallel_tempering(pt_ready)

        [sim.join() for sim in self.simulations]

    def _adjacent_temperature(self, pi: ParallelTemperingParameters):
        """Find the value of adjacent temperature within the values that are present in the simulations."""
        temp_index = self._temperatures.index(pi.parameters.temperature)
        if temp_index + 1 == len(self._temperatures):
            return self._temperatures[temp_index - 1]
        else:
            return self._temperatures[temp_index + 1]

    def _simulation_running_this_temperature(self, temperature: float):
        """
        Return the running simulation that currently has 'temperature' set as the temperature it is running at.

        If no such simulation can be found, return None.
        """
        sims_for_temp = [sim for sim in self.simulations if np.isclose(sim.temperature.value, temperature) and sim.running.value]
        if sims_for_temp:
            return sims_for_temp[0]
        else:
            return None

    def _do_parallel_tempering(self, pt_ready: List[ParallelTemperingParameters]):
        """Manage random selection of temperatures and exchanging parameters between configurations."""
        # process waiting list for parallel tempering in random order
        import random
        pt_param = None
        it = len(pt_ready)
        while it > 0 and pt_ready:
            pt_param = random.choice(pt_ready)
            adj_temp = self._adjacent_temperature(pt_param)
            if self._simulation_running_this_temperature(float(adj_temp)) is None:
                # unblock
                pt_param.pipe.send(pt_param.parameters)
                pt_ready.remove(pt_param)
                logger.debug(f'SimulationRunner: Freed up {pt_param} from PT waiting list')
            else:
                try:
                    adj_pt_param = [p for p in pt_ready if p.parameters.temperature == adj_temp][0]
                    if self._parallel_tempering_decision(pt_param, adj_pt_param):
                        # sending new parameters down the pipe will unblock waiting processes
                        pt_param.pipe.send(adj_pt_param.parameters)
                        adj_pt_param.pipe.send(pt_param.parameters)
                        logger.debug(f'SimulationRunner: Exchanged {pt_param.parameters.temperature} and {adj_pt_param.parameters.temperature}')
                    else:
                        # sending old parameters down the pipe will unblock waiting processes
                        pt_param.pipe.send(pt_param.parameters)
                        adj_pt_param.pipe.send(adj_pt_param.parameters)
                        logger.debug(f'SimulationRunner: Did not exchange {pt_param.parameters.temperature} and {adj_pt_param.parameters.temperature}')

                    pt_ready.remove(pt_param)
                    pt_ready.remove(adj_pt_param)
                except IndexError:
                    pass
            it -= 1

    @staticmethod
    def _parallel_tempering_decision(p1: ParallelTemperingParameters, p2: ParallelTemperingParameters) -> bool:
        t1, e1, _ = float(p1.parameters.temperature), p1.energy, p1.pipe
        t2, e2, _ = float(p2.parameters.temperature), p2.energy, p2.pipe
        d_b = 1 / t1 - 1 / t2
        d_e = e1 - e2
        return d_b * d_e > 0 or np.random.random() < np.exp(d_b * d_e)

    def stop(self):
        [sim.terminate() for sim in self.simulations]

    def alive(self):
        return self._starting or [sim for sim in self.simulations if sim.is_alive() or sim.running.value]
