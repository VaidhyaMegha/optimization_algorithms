import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optimization functions
def sphere_function(x):
    return np.sum(x**2)

# Initialization
def initialize_population(pop_size, dim, bounds):
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
    return population

# Ensure position is within bounds
def clip_position(position, bounds):
    return np.clip(position, bounds[0], bounds[1])

# Optimization Step
def optimization_step(population, global_best, global_worst, bounds, dim, objective_function):
    new_population = np.copy(population)
    u = bounds[1]
    l = bounds[0]
    x_mean = np.mean(population, axis=0)

    for i in range(len(population)):
        x_old = population[i]
        x_rand = population[np.random.randint(len(population))]
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        r3 = np.random.uniform(0, 1)
        r4 = np.random.uniform(0, 1)

        # Generate random perturbation
        R = u - ((u - l) * r3)
        T = np.random.choice([1, 2])

        if r4 > 0.5:
            new_position = x_old + r1 * (global_best - T * x_rand) - r2 * (global_worst - x_rand)
        else:
            new_position = R

        new_position = clip_position(new_position, bounds)
        if objective_function(new_position) < objective_function(x_old):
            new_population[i] = new_position

    return new_population

# Main optimization loop
def optimize(objective_function, dim, bounds, pop_size, max_iter, func_name, optimum_value):
    population = initialize_population(pop_size, dim, bounds)
    iteration_history = []
    population_history = []
    best_solution_history = []
    best_scores = []
    best_scores_per_iteration = []

    for t in range(max_iter):
        global_best = min(population, key=objective_function)
        global_worst = max(population, key=objective_function)
        population = optimization_step(population, global_best, global_worst, bounds, dim, objective_function)
        best_score = objective_function(global_best)
        best_scores.append(best_score)

        # Save iteration and population history
        iteration_info = {
            'iteration': t,
            'best_score': best_score,
        }
        iteration_history.append(iteration_info)
        population_history.append(population.copy())
        best_solution_history.append(global_best.copy())
        best_scores_per_iteration.append(best_score)

        # Check if optimum value is reached
        if best_score <= optimum_value:
            break

    global_best = min(population, key=objective_function)
    final_score = objective_function(global_best)
    optimization_history = {
        'function_name': func_name,
        'iteration_history': iteration_history,
        'population_history': population_history,
        'best_solution_history': best_solution_history,
        'final_best_solution': global_best,
        'final_best_score': final_score,
        'final_iteration': t+1,  # Final iteration number
        'best_scores_per_iteration': best_scores_per_iteration
    }

    return optimization_history

# List of functions with their dimensions, bounds, and optimum values
functions = [
    ("Sphere", sphere_function, 30, [-100, 100], 0),
]

# Execute optimization for each function with varying population sizes
for name, func, dim, bounds, opt_value in functions:
    print("---" * 50)
    all_best_scores = []
    all_best_populations = []
    all_mfe = []
    convergence_data = []

    for r in range(30):
        for i in range(10):
            pop_size = 10 * (i + 1)
            max_iter = (500000 // pop_size)
            bounds = [np.full(dim, bounds[0]), np.full(dim, bounds[1])] if isinstance(bounds[0], (int, float)) else bounds
            optimization_history = optimize(func, dim, bounds, pop_size=pop_size, max_iter=max_iter, func_name=name, optimum_value=opt_value)

            best_position = optimization_history['final_best_solution']
            best_score = optimization_history['final_best_score']
            final_iteration = optimization_history['final_iteration']

            mfe = pop_size * final_iteration

            all_best_scores.append(best_score)
            all_best_populations.append((pop_size, best_position))
            all_mfe.append(mfe)

            convergence_data.append(optimization_history['best_scores_per_iteration'])

            print(f"Run {r + 1}, Population Size {pop_size}, Best Score: {best_score}, FE: {mfe}")

            if best_score <= opt_value:
                break

    # Select best 30 scores and corresponding populations
    best_30_scores_indices = np.argsort(all_best_scores)[:30]
    best_30_scores = np.array(all_best_scores)[best_30_scores_indices]
    best_30_populations = [all_best_populations[i] for i in best_30_scores_indices]
    best_30_mfe = np.array(all_mfe)[best_30_scores_indices]

    # Perform statistics
    best_score = np.min(best_30_scores)
    worst_score = np.max(best_30_scores)
    mean_score = np.mean(best_30_scores)
    std_score = np.std(best_30_scores)
    mean_mfe = np.mean(best_30_mfe)

    # Save results
    os.makedirs(name, exist_ok=True)

    # Save best population history
    population_data = {
        'population_size': [],
        'best_population': []
    }
    for pop_size, pop in best_30_populations:
        population_data['population_size'].append(pop_size)
        population_data['best_population'].append(pop.tolist())

    population_df = pd.DataFrame(population_data)
    population_file_name = os.path.join(name, 'best_population_history.csv')
    population_df.to_csv(population_file_name, index=False)

    # Save best solution history
    best_solution_data = {
        'best_score': best_30_scores,
        'mfe': best_30_mfe
    }
    best_solution_df = pd.DataFrame(best_solution_data)
    best_solution_file_name = os.path.join(name, 'best_solution_history.csv')
    best_solution_df.to_csv(best_solution_file_name, index=False)

    # Save statistics
    stats_data = {
        'Function Name': [name],
        'Best Score': [best_score],
        'Worst Score': [worst_score],
        'Mean Score': [mean_score],
        'Standard Deviation': [std_score],
        'Mean MFE': [mean_mfe]
    }
    stats_df = pd.DataFrame(stats_data)
    stats_file_name = os.path.join(name, 'statistics.csv')
    stats_df.to_csv(stats_file_name, index=False)

    # Save best scores
    best_scores_df = pd.DataFrame({'Best Scores': best_30_scores})
    best_scores_file_name = os.path.join(name, 'best_scores.csv')
    best_scores_df.to_csv(best_scores_file_name, index=False)

    # Compute mean best scores per iteration
    max_iterations = max([len(run) for run in convergence_data])
    mean_best_scores_per_iteration = np.zeros(max_iterations)
    for i in range(max_iterations):
        iteration_scores = [run[i] for run in convergence_data if len(run) > i]
        mean_best_scores_per_iteration[i] = np.mean(iteration_scores)

    # Save convergence data
    convergence_df = pd.DataFrame({
        'Iteration': np.arange(max_iterations),
        'Mean Best Score': mean_best_scores_per_iteration
    })
    convergence_file_name = os.path.join(name, 'convergence_curve.csv')
    convergence_df.to_csv(convergence_file_name, index=False)

    # Plot convergence curve
    plt.figure()
    plt.plot(np.arange(max_iterations), mean_best_scores_per_iteration, label='Mean Best Score')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Best Score')
    plt.title(f'Convergence Curve for {name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(name, 'convergence_curve.png'))
    plt.close()
