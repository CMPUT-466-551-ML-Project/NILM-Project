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
        self.aggregate = TimeSeries()
        self.aggregate.array.resize(3)
        self.aggregate.array[0] = (np.uint32(1), np.float32(5.0))
        self.aggregate.array[1] = (np.uint32(2), np.float32(2.0))
        self.aggregate.array[2] = (np.uint32(3), np.float32(4.0))

        device1 = TimeSeries('a')
        device1.array.resize(3)
        device1.array[0] = (np.uint32(1), np.float32(5.0))
        device1.array[1] = (np.uint32(2), np.float32(5.0))
        device1.array[2] = (np.uint32(3), np.float32(5.0))

        device2 = TimeSeries('b')
        device2.array.resize(3)
        device2.array[0] = (np.uint32(1), np.float32(0.0))
        device2.array[1] = (np.uint32(2), np.float32(5.0))
        device2.array[2] = (np.uint32(3), np.float32(0.0))

        self.devices = [device1, device2]

        activations = [d.indicators(np.float32(0.0)) for d in self.devices]
        self.indicator_matrix = np.column_stack(activations)

    def test_sort_data(self):
        """Test the estimator by finding intervals of a single active device."""
        estimate = confidence_estimator(self.aggregate.powers, self.devices,
                                        self.indicator_matrix, sort_data)
        estimate_test = {'a': np.float32(4.5), 'b': np.float32(-2.5)}
        self.assertEqual(estimate, estimate_test)

    def test_get_changed_data(self):
        """
        Test the estimator by looking for immediate power changes from a single
        device changing state.
        """
        estimate = confidence_estimator(self.aggregate.powers, self.devices,
                                        self.indicator_matrix, get_changed_data)
        estimate_test = {'a': np.float(0.0), 'b': np.float(2.5)}
        self.assertEqual(estimate, estimate_test)
