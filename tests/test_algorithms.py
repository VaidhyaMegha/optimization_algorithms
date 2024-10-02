import unittest
import numpy as np
from rao_algorithms import BMR_algorithm, BWR_algorithm, objective_function, constraint_1, constraint_2

class TestOptimizationAlgorithms(unittest.TestCase):

    def setUp(self):
        self.bounds = np.array([[-100, 100]] * 2)
        self.num_iterations = 100
        self.population_size = 50
        self.num_variables = 2

    def test_bmr_unconstrained(self):
        best_solution, _ = BMR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, objective_function)
        self.assertIsInstance(best_solution, np.ndarray)

    def test_bwr_unconstrained(self):
        best_solution, _ = BWR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, objective_function)
        self.assertIsInstance(best_solution, np.ndarray)

    def test_bmr_constrained(self):
        constraints = [constraint_1, constraint_2]
        best_solution, _ = BMR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, objective_function, constraints)
        self.assertIsInstance(best_solution, np.ndarray)

    def test_bwr_constrained(self):
        constraints = [constraint_1, constraint_2]
        best_solution, _ = BWR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, objective_function, constraints)
        self.assertIsInstance(best_solution, np.ndarray)

if __name__ == '__main__':
    unittest.main()