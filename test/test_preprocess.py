"""
Unit tests for evaluation functions.
"""

# pylint: disable=E1101

import unittest

import numpy as np

from nilm.preprocess import solve_constant_energy
from nilm.timeseries import TimeSeries


class TestPreprocess(unittest.TestCase):
    """
    Test the evaluation functions.
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

        self.assertTrue(all([np.isclose(e[0], e[1], atol=np.float32(1e-9)) for
                                        e in zip(energies_test, energies)]))

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

        self.assertTrue(all([np.isclose(e[0], e[1], atol=np.float32(1e-3)) for
                                        e in zip(energies_test, energies)]))
