import numpy as np
from .penalty import constrained_objective_function

def BMR_algorithm(bounds, num_iterations, population_size, num_variables, objective_func, constraints=None):
    population = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(population_size, num_variables))

    best_scores = []

    for iteration in range(num_iterations):
        if constraints:
            fitness = [constrained_objective_function(ind, objective_func, constraints) for ind in population]
        else:
            fitness = np.apply_along_axis(objective_func, 1, population)

        best_solution = population[np.argmin(fitness)]
        mean_solution = np.mean(population, axis=0)
        best_score = np.min(fitness)
        best_scores.append(best_score)

        for i in range(population_size):
            r1, r2, r3, r4 = np.random.rand(4)
            T = np.random.choice([1, 2])
            random_solution = population[np.random.randint(population_size)]

            if r4 > 0.5:
                population[i] += r1 * (best_solution - T * mean_solution) + r2 * (best_solution - random_solution)
            else:
                population[i] = bounds[:, 1] - (bounds[:, 1] - bounds[:, 0]) * r3

        population = np.clip(population, bounds[:, 0], bounds[:, 1])

    return best_solution, best_scores


def BWR_algorithm(bounds, num_iterations, population_size, num_variables, objective_func, constraints=None):
    population = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(population_size, num_variables))

    best_scores = []

    for iteration in range(num_iterations):
        if constraints:
            fitness = [constrained_objective_function(ind, objective_func, constraints) for ind in population]
        else:
            fitness = np.apply_along_axis(objective_func, 1, population)

        best_solution = population[np.argmin(fitness)]
        worst_solution = population[np.argmax(fitness)]
        best_score = np.min(fitness)
        best_scores.append(best_score)

        for i in range(population_size):
            r1, r2, r3, r4 = np.random.rand(4)
            T = np.random.choice([1, 2])
            random_solution = population[np.random.randint(population_size)]

            if r4 > 0.5:
                population[i] += r1 * (best_solution - T * random_solution) - r2 * (worst_solution - random_solution)
            else:
                population[i] = bounds[:, 1] - (bounds[:, 1] - bounds[:, 0]) * r3

        population = np.clip(population, bounds[:, 0], bounds[:, 1])

    return best_solution, best_scores
