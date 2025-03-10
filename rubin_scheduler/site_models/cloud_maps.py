__all__ = ("CloudMap",)

import healpy as hp
import numpy as np

from rubin_scheduler.scheduler.utils import match_hp_resolution
from rubin_scheduler.utils import DEFAULT_NSIDE


class CloudMap(object):
    """A class to hold cloud maps over time

    Parameters
    ----------
    nside_out : `int`
        The nside to output maps as. Default to
        DEFAULT_NSIDE (probably 32).
    time_limit : `float`
        Do not return a cloud map if there is nothing within
        the time limit. Default 20 (minutes).
    """

    def __init__(self, nside_out=DEFAULT_NSIDE, time_limit=20.0, max_frames=2000):
        self.time_limit = time_limit / 60 / 24.0  # to days
        self.nside_out = nside_out
        self.mjds = []
        self.cloud_extinction_hparrays = []
        self.max_frames = max_frames

    def add_frame(self, input_cloud_extinction, mjd, nested=False):
        """Add a frame. Will be converted to ring order if needed
        and set to the self.nside_out resolution. Sets any values of
        negative extinction to zero.

        Parameters
        ----------
        input_cloud_extinction : `np.array`
            HEALpix array with
        mjd : `float`
           The MJD of the cloud frame
        nested : `bool`
            If True, converts the incoming map to ring order.
            Default False
        """
        self.mjds.append(mjd)

        to_add = input_cloud_extinction.copy()
        if nested:
            to_add = hp.reorder(to_add, n2r=True)

        # Set to the proper resolution.
        to_add = match_hp_resolution(to_add, self.nside_out)

        # We don't believe anything that has negative extinction
        too_clear = np.where(to_add < 0)[0]
        to_add[too_clear] = 0

        self.cloud_extinction_hparrays.append(to_add)

        # If things were entered out of order
        if not np.all(self.mjds[:-1] <= self.mjds[1:]):
            order = np.argsort(self.mjds)
            self.mjds = self.mjds[order]
            self.cloud_extinction_hparrays = self.cloud_extinction_hparrays[order]

        # If we are getting to the limit of out size
        while len(self.mjds) > self.max_frames:
            del self.max_frames[0]
            del self.mjds[0]

    def extinction_closest(self, mjd, hpid=None):
        """Return the closest map

        Parameters
        ----------
        mjd : `float`
            MJD of when the desired
        hpid : `int`
            If only a subset of the HEALpix array is needed, can
            specify index.

        Returns
        -------
        Healpix array in ring order with cloud extinction in magnitudes.

        """
        diff = np.abs(np.array(self.mjds) - mjd)
        # If we don't have a close enough map, just return zero
        if np.min(diff) > self.time_limit:
            return 0
        indx = np.min(np.where(diff == diff.min())[0])
        if hpid is not None:
            return self.cloud_extinction_hparrays[indx][hpid]
        return self.cloud_extinction_hparrays[indx]

    def extinction_forecast(self, mjd):
        """Spot to put in projected cloud extinction in the future"""
        raise NotImplementedError("Not done yet")
