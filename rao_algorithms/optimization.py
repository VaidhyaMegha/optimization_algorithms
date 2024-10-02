import numpy as np
import os
import csv

def initialize_population(bounds, population_size, num_variables):
    """Initialize population with random values within bounds."""
    return np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(population_size, num_variables))

def clip_position(position, bounds):
    """Clip the position to make sure it stays within bounds."""
    return np.clip(position, bounds[:, 0], bounds[:, 1])

def run_optimization(algorithm, bounds, num_iterations, population_size, num_variables, objective_function, constraints=None):
    """Run the selected algorithm and handle logging, saving results, etc."""
    
    # Initialize population and variables
    population = initialize_population(bounds, population_size, num_variables)
    best_scores = []

    # Prepare directory for saving results
    if not os.path.exists('results'):
        os.makedirs('results')

    # Run the algorithm
    best_solution, best_scores = algorithm(bounds, num_iterations, population_size, num_variables, objective_function, constraints)

    # Save results
    save_convergence_curve(best_scores)

    return best_solution, best_scores

def save_convergence_curve(best_scores):
    """Save the convergence curve as a CSV."""
    with open(f'results/convergence_curve.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Best Score'])
        for i, score in enumerate(best_scores):
            writer.writerow([i, score])
