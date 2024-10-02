import numpy as np

# Sphere function (from previous implementation)
def objective_function(x):
    return np.sum(x**2)

# Rastrigin function
def rastrigin_function(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Ackley function
def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    s1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
    s2 = -np.exp(np.sum(np.cos(c * x)) / n)
    return s1 + s2 + a + np.exp(1)

# Rosenbrock function
def rosenbrock_function(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

# Example constraint: x[0] + x[1] <= 10
def constraint_1(x):
    return x[0] + x[1] - 10  # Must be <= 0

# Example constraint: x[0] >= x[1]
def constraint_2(x):
    return -(x[0] - x[1])  # Must be <= 0 (x[0] >= x[1])

# Example: A nonlinear constrained optimization problem
def nonlinear_objective_function(x):
    return x[0]**2 + x[1]**2 + x[0]*x[1] - 10*x[0] - 15*x[1]

# Constraints
def constraint_3(x):
    return 25 - (x[0]**2 + x[1]**2)  # Circle constraint: x[0]^2 + x[1]^2 <= 25

def constraint_4(x):
    return x[0] - 5  # Must be <= 5

def constraint_5(x):
    return x[1] - 5  # Must be <= 5
