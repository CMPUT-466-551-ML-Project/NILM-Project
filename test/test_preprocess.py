"""
Unit tests for evaluation functions.
"""

# pylint: disable=E1101

import unittest

import numpy as np

from nilm.confidence_estimator import (confidence_estimator, get_changed_data,
                                       sort_data)
from nilm.preprocess import solve_constant_energy
from nilm.timeseries import TimeSeries


class TestPreprocessConstantEnergy(unittest.TestCase):
    """
    Test the constant energy preprocessing function.
    """
    def test_constant_energy_identity(self):
        """Test constant energy preprocessing on the identity matrix."""
        ts = [TimeSeries(), TimeSeries(), TimeSeries()]

        ts[0].array.resize(3)
        ts[0].array[0] = (np.uint32(1), np.float32(1.0))
        ts[0].array[1] = (np.uint32(2), np.float32(0.0))
        ts[0].array[2] = (np.uint32(3), np.float32(0.0))

        ts[1].array.resize(3)
        ts[1].array[0] = (np.uint32(1), np.float32(0.0))
        ts[1].array[1] = (np.uint32(2), np.float32(1.0))
        ts[1].array[2] = (np.uint32(3), np.float32(0.0))

        ts[2].array.resize(3)
        ts[2].array[0] = (np.uint32(1), np.float32(0.0))
        ts[2].array[1] = (np.uint32(2), np.float32(0.0))
        ts[2].array[2] = (np.uint32(3), np.float32(1.0))

        aggregated = ts[0] + ts[1] + ts[2]

        energies = solve_constant_energy(aggregated.powers,
                                         [t.indicators() for t in ts])

        energies_test = [np.float32(1.0), np.float32(1.0), np.float(1.0)]

        self.assertTrue(np.allclose(energies_test, energies[0],
                                    atol=np.float32(1e-3)))

    def test_constant_energy(self):
        """Test constant energy preprocessing on a more complicated matrix."""
        ts = [TimeSeries(), TimeSeries(), TimeSeries()]

        ts[0].array.resize(3)
        ts[0].array[0] = (np.uint32(1), np.float32(2.0))
        ts[0].array[1] = (np.uint32(2), np.float32(2.0))
        ts[0].array[2] = (np.uint32(3), np.float32(0.0))

        ts[1].array.resize(3)
        ts[1].array[0] = (np.uint32(1), np.float32(0.0))
        ts[1].array[1] = (np.uint32(2), np.float32(1.0))
        ts[1].array[2] = (np.uint32(3), np.float32(1.0))

        ts[2].array.resize(3)
        ts[2].array[0] = (np.uint32(1), np.float32(3.0))
        ts[2].array[1] = (np.uint32(2), np.float32(3.0))
        ts[2].array[2] = (np.uint32(3), np.float32(3.0))

        aggregated = ts[0] + ts[1] + ts[2]

        energies = solve_constant_energy(aggregated.powers,
                                         [t.indicators() for t in ts])

        energies_test = [np.float32(2.0), np.float32(1.0), np.float(3.0)]

        self.assertTrue(np.allclose(energies_test, energies[0],
                                    atol=np.float32(1e-3)))


class TestPreprocessConfidenceEstimator(unittest.TestCase):
    """
    Test the confidence estimator preprocessing functions.
    """

    def setUp(self):
        """Set up the data  needed for the confidence estimator."""
        self.time_series = {1:5, 2:2, 3:4}
        self.on_off = {(1, 'a'): True, (2, 'a'): True, (3, 'a'): True,
                       (1, 'b'): False, (2, 'b'): True, (3, 'b'): False}
        self.devices = ['a', 'b']

    def test_sort_data(self):
        """Test the estimator by finding intervals of a single active device."""
        estimate = confidence_estimator(self.time_series, self.on_off,
                                        self.devices, sort_data)
        estimate_test = {'a': 4, 'b': -2}
        self.assertEqual(estimate, estimate_test)

    def test_get_changed_data(self):
        """
        Test the estimator by looking for immediate power changes from a single
        device changing state.
        """
        estimate = confidence_estimator(self.time_series, self.on_off,
                                        self.devices, get_changed_data)
        estimate_test = {'a': 0, 'b': -2}
        self.assertEqual(estimate, estimate_test)
