from jmetal.core.solution import Solution


def is_feasible(solution):
    """
    Returns a boolean value concerning the feasibility of a solution
    :param solution:
    :return: true if the solution is feasible; false otherwise
    """
    return number_of_violated_constraints(solution) == 0


def number_of_violated_constraints(solution):
    """
    Returns the number of violated constraints of a solution
    :param solution:
    :return:
    """
    return sum([1 for _ in solution.constraints if _ < 0])


def overall_constraint_violation_degree(solution):
    """
    Returns the constraint violation degree of a solution, which is the sum of the constraint values that are not zero
    :param solution:
    :return:
    """
    return sum([value for value in solution.constraints if value < 0])


def feasibility_ratio(solutions):
    """
    Returns the percentage of feasible solutions in a solution list
    :param solutions:
    :return:
    """
    if len(solutions) == 0:
        raise Exception('The solution list is empty')
    return sum(1 for solution in solutions if is_feasible(solution)) / len(solutions)
