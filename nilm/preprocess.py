"""
A collection of data pre-processing algorithms
"""

import numpy as np

from scipy.optimize import minimize


def solve_constant_energy(aggregated, *device_activations):
    """
    Invert the indicator matrix, solving for the constant energy of each
    device.
    """
    def objective(power, total, matrix):
        """Objective function for the minimization."""
        return np.sum((total - np.dot(matrix, power)) ** 2)

    matrix = np.hstack((np.transpose(d) for d in device_activations))

    p0 = np.zeros(matrix.shape[1])
    bounds = [(0, np.inf)] * matrix.shape[1]

    solution = minimize(objective, p0, args=(aggregated, matrix),
                        method='SLSQP', bounds=bounds)

    return solution.x
