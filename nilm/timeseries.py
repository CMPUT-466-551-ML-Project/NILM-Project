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

    def indicators(self, threshold=np.float32(0.0)):
        """
        Returns the boolean on-off indicators for the timeseries, given a power
        threshold.
        """
        return np.apply_along_axis(lambda x: (x > threshold), 0, self.powers)

    def __add__(self, ts):
        """
        Add two timeseries together, based on the intersection of their
        timestamps.
        """
        indices1 = np.in1d(self.times, ts.times, assume_unique=True)
        indices2 = np.in1d(ts.times, self.times, assume_unique=True)

        ts_sum = TimeSeries()
        ts_sum.array = self.array[indices1]
        ts_sum.powers += ts.powers[indices2]

        return ts_sum

    def __sub__(self, ts):
        """
        Subtract two timeseries, based on the intersection of their timestamps.
        """
        indices1 = np.in1d(self.times, ts.times, assume_unique=True)
        indices2 = np.in1d(ts.times, self.times, assume_unique=True)

        ts_diff = TimeSeries()
        ts_diff.array = self.array[indices1]
        ts_diff.powers -= ts.powers[indices2]

        return ts_diff

    def intersect(self, ts):
        """
        Modify self to only contain the timestamps present in the given
        timeseries.
        """
        indices = np.in1d(self.times, ts.times, assume_unique=True)

        self.array = self.array[indices]
