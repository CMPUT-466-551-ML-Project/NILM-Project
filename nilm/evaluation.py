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
    array = zip(test.indicators(threshold), truth.indicators(threshold))

    tp = lambda x: 1 if (x[0] and x[1]) else 0
    fp = lambda x: 1 if (x[0] and not x[1]) else 0
    fn = lambda x: 1 if (not x[0] and x[1]) else 0

    true_positives = float(sum(tp(x) for x in array))
    false_positives = float(sum(fp(x) for x in array))
    false_negatives = float(sum(fn(x) for x in array))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return 2 * precision * recall / (precision + recall)


def mean_squared_error(test, truth):
    """Calculate the mean squared error between the two arrays."""
    return ((test - truth) ** 2).mean(axis=None)
