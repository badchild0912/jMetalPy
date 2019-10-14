from abc import ABCMeta, abstractmethod

from jmetal.util.point import IdealPoint

"""
.. module:: aggregative_function
   :platform: Unix, Windows
   :synopsis: Implementation of aggregative (scalarizing) functions.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benitez-Hidalgo <antonio.b@uma.es>
"""


class AggregativeFunction():
    __metaclass__ = ABCMeta

    @abstractmethod
    def compute(self, vector, weight_vector):
        pass

    @abstractmethod
    def update(self, vector):
        pass


class WeightedSum(AggregativeFunction):

    def compute(self, vector, weight_vector):
        return sum(map(lambda x, y: x * y, vector, weight_vector))

    def update(self, vector):
        pass


class Tschebycheff(AggregativeFunction):

    def __init__(self, dimension):
        self.ideal_point = IdealPoint(dimension)

    def compute(self, vector, weight_vector):
        max_fun = -1.0e+30

        for i in range(len(vector)):
            diff = abs(vector[i] - self.ideal_point.point[i])

            if weight_vector[i] == 0:
                feval = 0.0001 * diff
            else:
                feval = diff * weight_vector[i]

            if feval > max_fun:
                max_fun = feval

        return max_fun

    def update(self, vector):
        self.ideal_point.update(vector)
