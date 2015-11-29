"""
A collection of data pre-processing algorithms
"""

import numpy as np

from scipy.optimize import minimize

from nilm.evaluation import mean_squared_error


def solve_constant_energy(aggregated, device_activations):
    """
    Invert the indicator matrix, solving for the constant energy of each
    device. We return the constant power for each device, and the mean squared
    error.
    """
    def objective(power, total, matrix):
        """Objective function for the minimization."""
        return np.sum((total - np.dot(matrix, power)) ** 2)

    matrix = np.column_stack(device_activations)

    p0 = np.zeros(matrix.shape[1])
    bounds = [(0, np.inf)] * matrix.shape[1]

    solution = minimize(objective, p0, args=(aggregated, matrix),
                        method='SLSQP', bounds=bounds)

    error = mean_squared_error(np.dot(matrix, solution.x), aggregated)

    return (solution.x, error)

def confidence(data):
    """A Heuristic for how usable our current estimate of data is."""

    if len(data) == 0:
        return np.inf

    mean = data.mean(axis=None)
    variance = ((data - mean) ** 2).mean(axis=None)

    return variance / len(data)


def only_device(device_idx, time_idx, indicator_matrix):
    """
    Returns True if the device is the only device active at a certain time.
    """
    devices_on = np.where(indicator_matrix[time_idx, :])[0]

    return (device_idx in devices_on) and (len(devices_on) == 1)


def sort_data(aggregated, devices, indicator_matrix):
    """
    Generates usable samples for each device, where that
    device was the only device active at a single time period.
    """

    data = [[] for _ in xrange(len(devices))]

    for d in xrange(len(devices)):
        for t in xrange(len(devices[d].times)):
            if only_device(d, t, indicator_matrix):
                data[d].append(aggregated[t])

    return np.array([np.array(d) for d in data])


def changed_devices(devices, time_idx, indicator_matrix):
    """
    Returns all devices whose I/O state changed between at the given time
    period.
    """
    return [d for d in xrange(len(devices)) if
            (indicator_matrix[time_idx, d] != indicator_matrix[time_idx-1, d])]


def get_changed_data(aggregated, devices, indicator_matrix):
    """
    Generates data for each device by the step inference method, calculating
    the change in energy usage as a single device changes.
    """

    data = [[] for _ in xrange(len(devices))]

    for t in xrange(1, len(devices[0].times)):
        changed = changed_devices(devices, t, indicator_matrix)

        if len(changed) == 1:
            power_diff = abs(aggregated[t] - aggregated[t-1])
            data[changed[0]].append(power_diff)

    return np.array([np.array(d) for d in data])


def confidence_estimator(aggregated, devices, indicator_matrix, data_sorter):
    """
    Given a time series of aggregated data, time series of devices, and a matrix
    of on/off indicators, computes the best power estimators by the confidence
    interval subtraction method.  Data obtained for the confidence interval
    measure is obtained via the data_sorter function, which is currently
    implemented to either take samples from immediate changes in power from a
    single device switching on (get_changed_data) or taking samples from when a
    single device is on (sort_data). The program assumes that the time data
    between the series and indicators is the same, but does not assume how it
    is distributed. This function assumes that every device will be able to be
    calculated at some point. If not, this function is not able to estimate the
    programs accurately.
    """
    if len(devices) == 0:
        return {}

    data = data_sorter(aggregated, devices, indicator_matrix)

    # Pick data to remove according to some heuristic
    heuristic = lambda x: confidence(data[x])
    choice = min(range(len(devices)), key=heuristic)

    if heuristic(choice) == np.inf:
        # Need to pick a better approach, try
        # generating more data using level technique.
        mean_choice = np.float32(0.0)
    else:
        mean_choice = data[choice].mean(axis=None)

    new_aggregated = aggregated - indicator_matrix[:, choice] * mean_choice

    new_devices = devices[:choice] + devices[choice+1:]
    new_indicators = np.delete(indicator_matrix, choice, 1)
    calculated_means = confidence_estimator(new_aggregated, new_devices,
                                            new_indicators, data_sorter)
    calculated_means[devices[choice].name] = mean_choice

    return calculated_means
