"""
A collection of functions for evaluating TimeSeries.
"""

# pylint: disable=E1101

import numpy as np


def f_score(test, truth, threshold=np.float32(0.0)):
    """
    Calculate the F1 score of the test timeseries against the ground truth,
    with indicators calculated according to the given threshold.

    We assume both timeseries have the same length, and are aligned in time.
    """
    test_ind = test.indicators(threshold)
    test_ind_not = np.logical_not(test_ind)
    truth_ind = truth.indicators(threshold)
    truth_ind_not = np.logical_not(truth_ind)

    true_positives = np.sum(np.logical_and(test_ind, truth_ind),
                            dtype=np.float32)
    false_positives = np.sum(np.logical_and(test_ind, truth_ind_not),
                             dtype=np.float32)
    false_negatives = np.sum(np.logical_and(test_ind_not, truth_ind),
                             dtype=np.float32)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return 2 * precision * recall / (precision + recall)


def mean_squared_error(test, truth):
    """Calculate the mean squared error between the two arrays."""
    return ((test - truth) ** 2).mean(axis=None)
