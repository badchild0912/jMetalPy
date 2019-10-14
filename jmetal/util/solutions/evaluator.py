import functools
from abc import ABCMeta, abstractmethod
from multiprocessing.pool import ThreadPool, Pool
from typing import TypeVar, List, Generic

import dask
from pyspark import SparkConf, SparkContext

from jmetal.core.problem import Problem

S = TypeVar('S')


class Evaluator(Generic[S]):
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self, solution_list, problem):
        pass

    @staticmethod
    def evaluate_solution(solution, problem):
        problem.evaluate(solution)


class SequentialEvaluator(Evaluator[S]):

    def evaluate(self, solution_list, problem):
        for solution in solution_list:
            Evaluator.evaluate_solution(solution, problem)

        return solution_list


class MapEvaluator(Evaluator[S]):

    def __init__(self, processes = None):
        self.pool = ThreadPool(processes)

    def evaluate(self, solution_list, problem):
        self.pool.map(lambda solution: Evaluator.evaluate_solution(solution, problem), solution_list)

        return solution_list


class SparkEvaluator(Evaluator[S]):
    def __init__(self, processes = 8):
        self.spark_conf = SparkConf().setAppName("jMetalPy").setMaster("local[{processes}]")
        self.spark_context = SparkContext(conf=self.spark_conf)

        logger = self.spark_context._jvm.org.apache.log4j
        logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)

    def evaluate(self, solution_list, problem):
        solutions_to_evaluate = self.spark_context.parallelize(solution_list)

        return solutions_to_evaluate \
            .map(lambda s: problem.evaluate(s)) \
            .collect()


def evaluate_solution(solution, problem):
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class DaskEvaluator(Evaluator[S]):
    def __init__(self, scheduler='processes'):
        self.scheduler = scheduler

    def evaluate(self, solution_list, problem):
        with dask.config.set(scheduler=self.scheduler):
            return list(dask.compute(*[
                dask.delayed(evaluate_solution)(solution=solution, problem=problem) for solution in solution_list
            ]))


class MultiprocessEvaluator(Evaluator[S]):
    def __init__(self, processes=None):
        super().__init__()
        self.pool = Pool(processes)

    def evaluate(self, solution_list, problem):
        return self.pool.map(functools.partial(evaluate_solution, problem=problem), solution_list)

