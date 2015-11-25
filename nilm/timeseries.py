"""
Utility class for dealing with timeseries data.
"""

# pylint: disable=E1101

import numpy as np


class TimeSeries(object):
    """
    Object for holding, manipulating, and loading power timeseries data.
    """
    def __init__(self, path=None):
        if path is not None:
            self.array = np.genfromtxt(path, dtype=[('time', np.uint32),
                                                    ('power', np.float32)])
        else:
            self.array = np.rec.array((0, 2), dtype=[('time', np.uint32),
                                                     ('power', np.float32)])

    @property
    def times(self):
        """Returns the array of times in the series."""
        return self.array['time']

    @property
    def powers(self):
        """Returns the array of powers in the series."""
        return self.array['power']

    @powers.setter
    def powers(self, power_array):
        """Set the powers in the series to a copy of the given array-."""
        self.array['power'] = power_array.copy()
