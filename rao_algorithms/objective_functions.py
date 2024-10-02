import numpy as np

# Example of an objective function (Sphere function)
def objective_function(x):
    return np.sum(x**2)

# Example constraint: x[0] + x[1] <= 10
def constraint_1(x):
    return x[0] + x[1] - 10  # Must be <= 0

# Example constraint: x[0] >= x[1]
def constraint_2(x):
    return -(x[0] - x[1])  # Must be <= 0 (x[0] >= x[1])
