import unittest
import numpy as np
from rao_algorithms import BMR_algorithm, BWR_algorithm, objective_function, constraint_1, constraint_2

class TestOptimizationAlgorithms(unittest.TestCase):

    def setUp(self):
        self.bounds = np.array([[-100, 100]] * 2)  # For a 2D problem
        self.num_iterations = 100
        self.population_size = 50
        self.num_variables = 2

    def test_bmr_unconstrained(self):
        best_fitness = BMR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, objective_function)
        self.assertIsInstance(best_fitness, float)
        self.assertGreaterEqual(best_fitness, 0, "Fitness should be non-negative for the Sphere function")

    def test_bwr_unconstrained(self):
        best_fitness = BWR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, objective_function)
        self.assertIsInstance(best_fitness, float)
        self.assertGreaterEqual(best_fitness, 0, "Fitness should be non-negative for the Sphere function")

    def test_bmr_constrained(self):
        constraints = [constraint_1, constraint_2]
        best_fitness = BMR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, objective_function, constraints)
        self.assertIsInstance(best_fitness, float)
        self.assertGreaterEqual(best_fitness, 0, "Fitness should be non-negative even with constraints")

    def test_bwr_constrained(self):
        constraints = [constraint_1, constraint_2]
        best_fitness = BWR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, objective_function, constraints)
        self.assertIsInstance(best_fitness, float)
        self.assertGreaterEqual(best_fitness, 0, "Fitness should be non-negative even with constraints")

if __name__ == '__main__':
    unittest.main()
