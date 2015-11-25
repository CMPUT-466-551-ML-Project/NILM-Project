"""
Unit tests for TimeSeries.
"""

# pylint: disable=E1101

import unittest
import os

import numpy as np

from nilm.timeseries import TimeSeries


DATA_PATH = os.path.join(os.path.dirname(__file__), 'test.dat')


class TestTimeSeries(unittest.TestCase):
    """
    Test the TimeSeries class.
    """
    def test_creation(self):
        """Test TimeSeries creation from a file."""
        ts_file = TimeSeries(DATA_PATH)

        ts_test = TimeSeries()
        ts_test.array.resize(3)
        ts_test.array[0] = (np.uint32(1), np.float32(0.0))
        ts_test.array[1] = (np.uint32(2), np.float32(1.0))
        ts_test.array[2] = (np.uint32(3), np.float32(0.0))

        self.assertItemsEqual(ts_file.times, ts_test.times)
        self.assertItemsEqual(ts_file.powers, ts_test.powers)
