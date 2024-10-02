def penalty_function(x, constraints):
    penalty = 0.0
    for constraint in constraints:
        constraint_value = constraint(x)
        if constraint_value > 0:
            penalty += constraint_value ** 2  # Penalize constraint violation
    return penalty

def constrained_objective_function(x, objective_func, constraints):
    obj_value = objective_func(x)
    penalty = penalty_function(x, constraints)
    return obj_value + penalty
