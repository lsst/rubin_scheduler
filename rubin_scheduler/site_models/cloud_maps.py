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
        self.cloud_extinction_uncerts = []
        self.max_frames = max_frames

    def add_frame(self, input_cloud_extinction, mjd, nested=False, uncert=None):
        """Add a frame. Will be converted to ring order if needed
        and set to the self.nside_out resolution.

        Parameters
        ----------
        input_cloud_extinction : `np.array`
            HEALpix array with extinction in mags.
        mjd : `float`
           The MJD of the cloud frame
        nested : `bool`
            If True, converts the incoming map to ring order.
            Default False
        uncert : `np.array`
            HEALpix array with uncertainty in extinction magnitudes.
        """
        self.mjds.append(mjd)

        to_add = input_cloud_extinction.copy()
        if uncert is None:
            to_add_uncert = to_add * 0
        else:
            to_add_uncert = uncert.copy()

        if nested:
            to_add = hp.reorder(to_add, n2r=True)
            to_add_uncert = hp.reorder(to_add_uncert, n2r=True)

        # Set to the proper resolution.
        to_add = match_hp_resolution(to_add, self.nside_out)
        to_add_uncert = match_hp_resolution(to_add_uncert, self.nside_out)

        self.cloud_extinction_hparrays.append(to_add)
        self.cloud_extinction_uncerts.append(to_add_uncert)

        # If things were entered out of order
        if not np.all(self.mjds[:-1] <= self.mjds[1:]):
            order = np.argsort(self.mjds)
            self.mjds = self.mjds[order]
            self.cloud_extinction_hparrays = self.cloud_extinction_hparrays[order]
            self.cloud_extinction_uncerts = self.cloud_extinction_uncerts[order]

        # If we are getting to the limit of out size
        while len(self.mjds) > self.max_frames:
            del self.cloud_extinction_hparrays[0]
            del self.cloud_extinction_uncerts[0]
            del self.mjds[0]

    def extinction_closest(self, mjd, hpid=None, uncert=False):
        """Return the closest map

        Parameters
        ----------
        mjd : `float`
            MJD of the desired extinction map.
        hpid : `int`
            If only a subset of the HEALpix array is needed, can
            specify index.
        uncert : `bool`
            Also return the uncertainty map with the extinction map.

        Returns
        -------
        Healpix array in ring order with cloud extinction in magnitudes.

        """
        diff = np.abs(np.array(self.mjds) - mjd)
        # If we don't have a close enough map, just return zero
        if np.min(diff) > self.time_limit:
            if uncert:
                return 0, 0
            else:
                return 0
        indx = np.min(np.where(diff == diff.min())[0])
        if hpid is not None:
            if uncert:
                return self.cloud_extinction_hparrays[indx][hpid], self.cloud_extinction_uncerts[indx][hpid]
            else:
                return self.cloud_extinction_hparrays[indx][hpid]

        if uncert:
            return self.cloud_extinction_hparrays[indx], self.cloud_extinction_uncerts[indx]
        else:
            return self.cloud_extinction_hparrays[indx]

    def extinction_forecast(self, mjd):
        """Spot to put in projected cloud extinction in the future"""
        raise NotImplementedError("Not done yet")
