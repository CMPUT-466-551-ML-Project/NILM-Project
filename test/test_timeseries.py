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
        ts_file = TimeSeries(path=DATA_PATH)

        ts_test = TimeSeries()
        ts_test.array.resize(3)
        ts_test.array[0] = (np.uint32(1), np.float32(0.0))
        ts_test.array[1] = (np.uint32(2), np.float32(1.0))
        ts_test.array[2] = (np.uint32(3), np.float32(0.0))

        self.assertItemsEqual(ts_file.times, ts_test.times)
        self.assertItemsEqual(ts_file.powers, ts_test.powers)

    def test_indicators(self):
        """Test getting the on-off indicators from a TimeSeries."""
        ts = TimeSeries()
        ts.array.resize(3)
        ts.array[0] = (np.uint32(1), np.float32(0.0))
        ts.array[1] = (np.uint32(2), np.float32(1.0))
        ts.array[2] = (np.uint32(3), np.float32(0.0))

        indicators_test = np.array([False, True, False])
        indicators = ts.indicators()

        self.assertItemsEqual(indicators, indicators_test)

    def test_indicators_threshold(self):
        """Test getting the on-off indicators with a threshold."""
        ts = TimeSeries()
        ts.array.resize(3)
        ts.array[0] = (np.uint32(1), np.float32(0.0))
        ts.array[1] = (np.uint32(2), np.float32(1.0))
        ts.array[2] = (np.uint32(3), np.float32(2.0))

        indicators_test = np.array([False, False, True])
        indicators = ts.indicators(np.float32(1.1))

        self.assertItemsEqual(indicators, indicators_test)

    def test_addition(self):
        """Test adding two TimeSeries together."""
        ts1 = TimeSeries()
        ts1.array.resize(3)
        ts1.array[0] = (np.uint32(1), np.float32(0.5))
        ts1.array[1] = (np.uint32(2), np.float32(1.0))
        ts1.array[2] = (np.uint32(3), np.float32(2.0))

        ts2 = TimeSeries()
        ts2.array.resize(3)
        ts2.array[0] = (np.uint32(1), np.float32(0.0))
        ts2.array[1] = (np.uint32(2), np.float32(1.0))
        ts2.array[2] = (np.uint32(3), np.float32(3.0))

        ts_test = TimeSeries()
        ts_test.array.resize(3)
        ts_test.array[0] = (np.uint32(1), np.float32(0.5))
        ts_test.array[1] = (np.uint32(2), np.float32(2.0))
        ts_test.array[2] = (np.uint32(3), np.float32(5.0))

        ts_add = ts1 + ts2

        self.assertItemsEqual(ts_add.powers, ts_test.powers)

    def test_addition_no_intersect(self):
        """ Test adding two TimeSeries with different timestamps."""
        ts1 = TimeSeries()
        ts1.array.resize(3)
        ts1.array[0] = (np.uint32(1), np.float32(0.5))
        ts1.array[1] = (np.uint32(2), np.float32(1.0))
        ts1.array[2] = (np.uint32(4), np.float32(2.0))

        ts2 = TimeSeries()
        ts2.array.resize(3)
        ts2.array[0] = (np.uint32(1), np.float32(0.0))
        ts2.array[1] = (np.uint32(3), np.float32(1.0))
        ts2.array[2] = (np.uint32(5), np.float32(3.0))

        ts_test = TimeSeries()
        ts_test.array.resize(1)
        ts_test.array[0] = (np.uint32(1), np.float32(0.5))

        ts_add = ts1 + ts2

        self.assertItemsEqual(ts_add.powers, ts_test.powers)

    def test_subtraction(self):
        """Test subtracting one TimeSeries from another."""
        ts1 = TimeSeries()
        ts1.array.resize(3)
        ts1.array[0] = (np.uint32(1), np.float32(0.5))
        ts1.array[1] = (np.uint32(2), np.float32(1.0))
        ts1.array[2] = (np.uint32(3), np.float32(3.0))

        ts2 = TimeSeries()
        ts2.array.resize(3)
        ts2.array[0] = (np.uint32(1), np.float32(0.0))
        ts2.array[1] = (np.uint32(2), np.float32(1.0))
        ts2.array[2] = (np.uint32(3), np.float32(2.0))

        ts_test = TimeSeries()
        ts_test.array.resize(3)
        ts_test.array[0] = (np.uint32(1), np.float32(0.5))
        ts_test.array[1] = (np.uint32(2), np.float32(0.0))
        ts_test.array[2] = (np.uint32(3), np.float32(1.0))

        ts_diff = ts1 - ts2

        self.assertItemsEqual(ts_diff.powers, ts_test.powers)

    def test_subtraction_no_intersect(self):
        """
        Test subtracting one TimeSeries from another with different timestamps.
        """
        ts1 = TimeSeries()
        ts1.array.resize(3)
        ts1.array[0] = (np.uint32(1), np.float32(0.5))
        ts1.array[1] = (np.uint32(2), np.float32(1.0))
        ts1.array[2] = (np.uint32(4), np.float32(3.0))

        ts2 = TimeSeries()
        ts2.array.resize(3)
        ts2.array[0] = (np.uint32(1), np.float32(0.0))
        ts2.array[1] = (np.uint32(3), np.float32(1.0))
        ts2.array[2] = (np.uint32(5), np.float32(2.0))

        ts_test = TimeSeries()
        ts_test.array.resize(1)
        ts_test.array[0] = (np.uint32(1), np.float32(0.5))

        ts_diff = ts1 - ts2

        self.assertItemsEqual(ts_diff.powers, ts_test.powers)

    def test_padding(self):
        """
        Test padding a TimeSeries with missing values.
        """
        ts_missing = TimeSeries()
        ts_missing.array.resize(3)
        ts_missing.array[0] = (np.uint32(1), np.float32(1.0))
        ts_missing.array[1] = (np.uint32(3), np.float32(3.0))
        ts_missing.array[2] = (np.uint32(5), np.float32(0.0))

        ts_test = TimeSeries()
        ts_test.array.resize(5)
        ts_test.array[0] = (np.uint32(1), np.float32(1.0))
        ts_test.array[1] = (np.uint32(2), np.float32(1.0))
        ts_test.array[2] = (np.uint32(3), np.float32(3.0))
        ts_test.array[3] = (np.uint32(4), np.float32(3.0))
        ts_test.array[4] = (np.uint32(5), np.float32(0.0))

        ts_missing.pad(5)

        self.assertItemsEqual(ts_missing.powers, ts_test.powers)

    def test_padding_gap(self):
        """
        Test padding a TimeSeries with a large gap.
        """
        ts_missing = TimeSeries()
        ts_missing.array.resize(3)
        ts_missing.array[0] = (np.uint32(1), np.float32(1.0))
        ts_missing.array[1] = (np.uint32(3), np.float32(3.0))
        ts_missing.array[2] = (np.uint32(8), np.float32(0.0))

        ts_test = TimeSeries()
        ts_test.array.resize(4)
        ts_test.array[0] = (np.uint32(1), np.float32(1.0))
        ts_test.array[1] = (np.uint32(2), np.float32(1.0))
        ts_test.array[2] = (np.uint32(3), np.float32(3.0))
        ts_test.array[3] = (np.uint32(8), np.float32(0.0))

        ts_missing.pad(3)

        self.assertItemsEqual(ts_missing.powers, ts_test.powers)

    def test_activations(self):
        """
        Test retrieving the TimeSeries activation indices.
        """
        ts = TimeSeries()
        ts.array.resize(11)
        ts.array[0] = (np.uint32(1), np.float32(1.0))
        ts.array[1] = (np.uint32(2), np.float32(1.0))
        ts.array[2] = (np.uint32(3), np.float32(3.0))
        ts.array[3] = (np.uint32(4), np.float32(3.0))
        ts.array[4] = (np.uint32(5), np.float32(0.0))
        ts.array[5] = (np.uint32(6), np.float32(1.0))
        ts.array[6] = (np.uint32(7), np.float32(1.0))
        ts.array[7] = (np.uint32(8), np.float32(0.0))
        ts.array[8] = (np.uint32(9), np.float32(3.0))
        ts.array[9] = (np.uint32(10), np.float32(0.0))
        ts.array[10] = (np.uint32(10), np.float32(1.0))

        test = [(0, 4), (5, 7), (8, 9), (10, 11)]
        activations = ts.activations(np.float32(0.5))

        self.assertItemsEqual(activations, test)
