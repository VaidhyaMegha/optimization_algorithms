import unittest
import numpy as np
from rao_algorithms import BMR_algorithm, BWR_algorithm, run_optimization, objective_function, rastrigin_function, ackley_function, rosenbrock_function, constraint_1, constraint_2

class TestOptimizationAlgorithms(unittest.TestCase):

    def setUp(self):
        self.bounds = np.array([[-100, 100]] * 2)  # Change as needed for higher dimensional problems
        self.num_iterations = 100
        self.population_size = 50
        self.num_variables = 2  # You can increase this for higher-dimensional tests

    def test_bmr_unconstrained(self):
        best_solution, _ = BMR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, objective_function)
        self.assertIsInstance(best_solution, np.ndarray)

    def test_bwr_unconstrained(self):
        best_solution, _ = BWR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, objective_function)
        self.assertIsInstance(best_solution, np.ndarray)

    def test_bmr_rastrigin(self):
        best_solution, _ = BMR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, rastrigin_function)
        self.assertIsInstance(best_solution, np.ndarray)

    def test_bwr_ackley(self):
        best_solution, _ = BWR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, ackley_function)
        self.assertIsInstance(best_solution, np.ndarray)

    def test_bmr_rosenbrock(self):
        best_solution, _ = BMR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, rosenbrock_function)
        self.assertIsInstance(best_solution, np.ndarray)

    def test_bmr_constrained(self):
        constraints = [constraint_1, constraint_2]
        best_solution, _ = BMR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, objective_function, constraints)
        self.assertIsInstance(best_solution, np.ndarray)

    def test_bwr_constrained(self):
        constraints = [constraint_1, constraint_2]
        best_solution, _ = BWR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, objective_function, constraints)
        self.assertIsInstance(best_solution, np.ndarray)

    def test_multiple_runs(self):
        """Run BMR multiple times and calculate mean and standard deviation."""
        num_runs = 30
        results = []

        for _ in range(num_runs):
            best_solution, _ = BMR_algorithm(self.bounds, self.num_iterations, self.population_size, self.num_variables, objective_function)
            results.append(np.sum(best_solution**2))  # Sum of squares for comparison

        mean_result = np.mean(results)
        std_result = np.std(results)

        self.assertGreaterEqual(mean_result, 0)
        print(f"BMR Mean: {mean_result}, Std Dev: {std_result}")

if __name__ == '__main__':
    unittest.main()
