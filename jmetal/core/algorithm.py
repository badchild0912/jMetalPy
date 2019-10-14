import logging
import threading
import time
from abc import abstractmethod, ABCMeta
from typing import TypeVar, Generic, List

from jmetal.config import store
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution

LOGGER = logging.getLogger('jmetal')

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: algorithm
   :platform: Unix, Windows
   :synopsis: Templates for algorithms.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benitez-Hidalgo <antonio.b@uma.es>
"""


class Algorithm(Generic[S, R], threading.Thread):
    __metaclass__ = ABCMeta

    def __init__(self):
        threading.Thread.__init__(self)

        self.solutions = []
        self.evaluations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0

        self.observable = store.default_observable

    @abstractmethod
    def create_initial_solutions(self):
        """ Creates the initial list of solutions of a metaheuristic. """
        pass

    @abstractmethod
    def evaluate(self, solution_list):
        """ Evaluates a solution list. """
        pass

    @abstractmethod
    def init_progress(self):
        """ Initialize the algorithm. """
        pass

    @abstractmethod
    def stopping_condition_is_met(self):
        """ The stopping condition is met or not. """
        pass

    @abstractmethod
    def step(self):
        """ Performs one iteration/step of the algorithm's loop. """
        pass

    @abstractmethod
    def update_progress(self):
        """ Update the progress after each iteration. """
        pass

    @abstractmethod
    def get_observable_data(self):
        """ Get observable data, with the information that will be send to all observers each time. """
        pass

    def run(self):
        """ Execute the algorithm. """
        self.start_computing_time = time.time()

        self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)

        LOGGER.debug('Initializing progress')
        self.init_progress()

        LOGGER.debug('Running main loop until termination criteria is met')
        while not self.stopping_condition_is_met():
            self.step()
            self.update_progress()

        self.total_computing_time = time.time() - self.start_computing_time

    @abstractmethod
    def get_result(self):
        pass

    @abstractmethod
    def get_name(self):
        pass


class DynamicAlgorithm(Algorithm[S, R]):
    __metaclass__ = ABCMeta

    @abstractmethod
    def restart(self):
        pass


class EvolutionaryAlgorithm(Algorithm[S, R]):
    __metaclass__ = ABCMeta

    def __init__(self,
                 problem,
                 population_size,
                 offspring_population_size):
        super(EvolutionaryAlgorithm, self).__init__()
        self.problem = problem
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size

    @abstractmethod
    def selection(self, population):
        """ Select the best-fit individuals for reproduction (parents). """
        pass

    @abstractmethod
    def reproduction(self, population):
        """ Breed new individuals through crossover and mutation operations to give birth to offspring. """
        pass

    @abstractmethod
    def replacement(self, population, offspring_population):
        """ Replace least-fit population with new individuals. """
        pass

    def get_observable_data(self):
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    def init_progress(self):
        self.evaluations = self.population_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def step(self):
        mating_population = self.selection(self.solutions)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate(offspring_population)

        self.solutions = self.replacement(self.solutions, offspring_population)

    def update_progress(self):
        self.evaluations += self.offspring_population_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    @property
    def label(self):
        return '{self.get_name()}.{self.problem.get_name()}'


class ParticleSwarmOptimization(Algorithm[FloatSolution, List[FloatSolution]]):
    __metaclass__ = ABCMeta

    def __init__(self,
                 problem,
                 swarm_size):
        super(ParticleSwarmOptimization, self).__init__()
        self.problem = problem
        self.swarm_size = swarm_size

    @abstractmethod
    def initialize_velocity(self, swarm):
        pass

    @abstractmethod
    def initialize_particle_best(self, swarm):
        pass

    @abstractmethod
    def initialize_global_best(self, swarm):
        pass

    @abstractmethod
    def update_velocity(self, swarm):
        pass

    @abstractmethod
    def update_particle_best(self, swarm):
        pass

    @abstractmethod
    def update_global_best(self, swarm):
        pass

    @abstractmethod
    def update_position(self, swarm):
        pass

    @abstractmethod
    def perturbation(self, swarm):
        pass

    def get_observable_data(self):
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    def init_progress(self):
        self.evaluations = self.swarm_size

        self.initialize_velocity(self.solutions)
        self.initialize_particle_best(self.solutions)
        self.initialize_global_best(self.solutions)

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def step(self):
        self.update_velocity(self.solutions)
        self.update_position(self.solutions)
        self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)

    def update_progress(self):
        self.evaluations += self.swarm_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    @property
    def label(self):
        return '{self.get_name()}.{self.problem.get_name()}'
