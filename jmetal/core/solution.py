from abc import ABCMeta
from typing import List, Generic, TypeVar

BitSet = List[bool]
S = TypeVar('S')


class Solution(Generic[S]):
    """ Class representing solutions """
    __metaclass__ = ABCMeta

    def __init__(self, number_of_variables, number_of_objectives, number_of_constraints= 0):
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constrains = number_of_constraints
        self.variables = [[] for _ in range(self.number_of_variables)]
        self.objectives = [0.0 for _ in range(self.number_of_objectives)]
        self.constraints = [0.0 for _ in range(self.number_of_constrains)]
        self.attributes = {}

    def __eq__(self, solution):
        if isinstance(solution, self.__class__):
            return self.variables == solution.variables
        return False

    def __str__(self):
        return 'Solution(variables={},objectives={},constraints={})'.format(self.variables, self.objectives, self.constraints)


class BinarySolution(Solution[BitSet]):
    """ Class representing float solutions """

    def __init__(self, number_of_variables, number_of_objectives, number_of_constraints = 0):
        super(BinarySolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)

    def __copy__(self):
        new_solution = BinarySolution(
            self.number_of_variables,
            self.number_of_objectives)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution

    def get_total_number_of_bits(self):
        total = 0
        for var in self.variables:
            total += len(var)

        return total

    def get_binary_string(self):
        string = ""
        for bit in self.variables[0]:
            string += '1' if bit else '0'
        return string


class FloatSolution(Solution[float]):
    """ Class representing float solutions """

    def __init__(self, lower_bound, upper_bound, number_of_objectives,
                 number_of_constraints = 0):
        super(FloatSolution, self).__init__(len(lower_bound), number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constrains)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.attributes = self.attributes.copy()

        new_solution.attributes = self.attributes.copy()

        return new_solution


class IntegerSolution(Solution[int]):
    """ Class representing integer solutions """

    def __init__(self, lower_bound, upper_bound, number_of_objectives,
                  number_of_constraints= 0):
        super(IntegerSolution, self).__init__(len(lower_bound), number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = IntegerSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constrains)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution


class PermutationSolution(Solution):
    """ Class representing permutation solutions """

    def __init__(self, number_of_variables, number_of_objectives, number_of_constraints = 0):
        super(PermutationSolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)

    def __copy__(self):
        new_solution = PermutationSolution(
            self.number_of_variables,
            self.number_of_objectives)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution
