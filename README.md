# Optimization Algorithms: BMR and BWR

This package implements two simple yet powerful optimization algorithms:
- **BMR (Best-Mean-Random) Algorithm**
- **BWR (Best-Worst-Random) Algorithm**

These algorithms are designed to solve both **constrained** and **unconstrained** optimization problems without relying on metaphors or algorithm-specific parameters. The package is based on the paper:

**Ravipudi Venkata Rao and Ravikumar Shah (2024)**, "BMR and BWR: Two simple metaphor-free optimization algorithms for solving real-life non-convex constrained and unconstrained problems." [arXiv:2407.11149v2](https://arxiv.org/abs/2407.11149).

## Features

- **Metaphor-Free**: No reliance on nature-inspired metaphors.
- **Simple**: No algorithm-specific parameters to tune.
- **Flexible**: Handles both constrained and unconstrained optimization problems.
  
## Installation

You can install this package directly from PyPI:

```bash
pip install rao_algorithms
```

Alternatively, you can clone this repository and install it locally:

```bash
git clone https://github.com/VaidhyaMegha/optimization_algorithms.git
cd optimization_algorithms
pip install .
```

## How to Use

### Example: Constrained BMR Algorithm

```python
import numpy as np
from rao_algorithms import run_optimization, BMR_algorithm, objective_function, constraint_1, constraint_2

# Constrained BMR
# ---------------
bounds = np.array([[-100, 100]] * 2)
num_iterations = 100
population_size = 50
constraints = [constraint_1, constraint_2]

best_solution, best_scores = run_optimization(BMR_algorithm, bounds, num_iterations, population_size, 2, objective_function, constraints)
print(f"Constrained BMR Best solution: {best_solution}")

```

### Example: Unconstrained BMR Algorithm

```python
import numpy as np
from rao_algorithms import BMR_algorithm, objective_function

# Unconstrained BMR
# ---------------
# Define the bounds for a 2D problem
bounds = np.array([[-100, 100]] * 2)

# Set parameters
num_iterations = 100
population_size = 50
num_variables = 2

# Run the BMR algorithm
best_solution, best_scores = BMR_algorithm(bounds, num_iterations, population_size, num_variables, objective_function)
print(f"Unconstrained BMR Best solution found: {best_solution}")
```

### Example: Constrained BWR Algorithm

```python
import numpy as np
from rao_algorithms import BWR_algorithm, objective_function, constraint_1, constraint_2

# Constrained BWR
# ---------------
# Define the bounds for a 2D problem
bounds = np.array([[-100, 100]] * 2)

# Set parameters
num_iterations = 100
population_size = 50
num_variables = 2
constraints = [constraint_1, constraint_2]

# Run the BWR algorithm with constraints
best_solution, best_scores = BWR_algorithm(bounds, num_iterations, population_size, num_variables, objective_function, constraints)
print(f"Constrained BWR Best solution found: {best_solution}")
```

### Unit Testing

This package comes with unit tests. To run the tests:

```bash
python -m unittest discover -s tests
```

You can also run the tests using Docker:

```bash
docker build -t optimization-algorithms .
docker run -it optimization-algorithms
```

## Algorithms Overview

### BMR (Best-Mean-Random) Algorithm

The BMR algorithm is based on the best, mean, and random solutions from the population. It works by updating solutions based on their interaction with these key elements.

- **Paper Citation**: R. V. Rao, R. Shah, *BMR and BWR: Two simple metaphor-free optimization algorithms*. [arXiv:2407.11149v2](https://arxiv.org/abs/2407.11149).

### BWR (Best-Worst-Random) Algorithm

The BWR algorithm updates solutions by considering the best, worst, and random solutions in the population. The algorithm balances exploration and exploitation through these interactions.

- **Paper Citation**: R. V. Rao, R. Shah, *BMR and BWR: Two simple metaphor-free optimization algorithms*. [arXiv:2407.11149v2](https://arxiv.org/abs/2407.11149).

## Docker Support

You can use the included `Dockerfile` to build and test the package quickly. To build and run the package in Docker:

```bash
docker build -t optimization-algorithms .
docker run -it optimization-algorithms
```

## License

This package is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## References

1. Ravipudi Venkata Rao, Ravikumar Shah, "BMR and BWR: Two simple metaphor-free optimization algorithms for solving real-life non-convex constrained and unconstrained problems," [arXiv:2407.11149v2](https://arxiv.org/abs/2407.11149).