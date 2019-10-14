from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, List

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: Operator
   :platform: Unix, Windows
   :synopsis: Templates for operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benitez-Hidalgo <antonio.b@uma.es>
"""


class Operator(Generic[S, R]):
    """ Class representing operator """
    __metaclass__ = ABCMeta

    @abstractmethod
    def execute(self, source):
        pass

    @abstractmethod
    def get_name(self):
        pass


def check_valid_probability_value(func):
    def func_wrapper(self, probability):
        if probability > 1.0:
            raise Exception('The probability is greater than one: {}'.format(probability))
        elif probability < 0.0:
            raise Exception('The probability is lower than zero: {}'.format(probability))

        res = func(self, probability)
        return res
    return func_wrapper


class Mutation(Operator[S, S]):
    """ Class representing mutation operator. """
    __metaclass__ = ABCMeta

    @check_valid_probability_value
    def __init__(self, probability):
        self.probability = probability


class Crossover(Operator[List[S], List[R]]):
    """ Class representing crossover operator. """
    __metaclass__ = ABCMeta

    @check_valid_probability_value
    def __init__(self, probability):
        self.probability = probability

    @abstractmethod
    def get_number_of_parents(self):
        pass

    @abstractmethod
    def get_number_of_children(self):
        pass


class Selection(Operator[S, R]):
    """ Class representing selection operator. """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass
