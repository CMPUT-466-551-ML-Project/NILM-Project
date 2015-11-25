"""
Unit tests for evaluation functions.
"""

# pylint: disable=E1101

import unittest

import numpy as np

from nilm.evaluation import f_score
from nilm.timeseries import TimeSeries


class TestEvaluationFunctions(unittest.TestCase):
    """
    Test the evaluation functions.
    """
    def test_f_score_perfect(self):
        """Test F1 score evaluation with no false positives or negatives."""
        ts_test = TimeSeries()
        ts_test.array.resize(5)
        ts_test.array[0] = (np.uint32(1), np.float32(0.0))
        ts_test.array[1] = (np.uint32(2), np.float32(1.0))
        ts_test.array[2] = (np.uint32(3), np.float32(1.0))
        ts_test.array[3] = (np.uint32(4), np.float32(1.0))
        ts_test.array[4] = (np.uint32(5), np.float32(0.0))

        score = f_score(ts_test, ts_test, threshold=np.float32(0.5))

        self.assertEqual(score, 1)

    def test_f_score(self):
        """Test F1 score evaluation with some false positives and negatives."""
        ts_test = TimeSeries()
        ts_test.array.resize(5)
        ts_test.array[0] = (np.uint32(1), np.float32(1.0))
        ts_test.array[1] = (np.uint32(2), np.float32(1.0))
        ts_test.array[2] = (np.uint32(3), np.float32(1.0))
        ts_test.array[3] = (np.uint32(4), np.float32(1.0))
        ts_test.array[4] = (np.uint32(5), np.float32(0.0))

        ts_truth = TimeSeries()
        ts_truth.array.resize(5)
        ts_truth.array[0] = (np.uint32(1), np.float32(0.0))
        ts_truth.array[1] = (np.uint32(2), np.float32(1.0))
        ts_truth.array[2] = (np.uint32(3), np.float32(1.0))
        ts_truth.array[3] = (np.uint32(4), np.float32(1.0))
        ts_truth.array[4] = (np.uint32(5), np.float32(1.0))

        score = f_score(ts_test, ts_truth, threshold=np.float32(0.5))

        self.assertEqual(score, 0.75)
