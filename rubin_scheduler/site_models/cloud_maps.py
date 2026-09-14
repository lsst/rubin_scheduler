__all__ = ("CloudMap",)

import healpy as hp
import numpy as np

from rubin_scheduler.utils import DEFAULT_NSIDE, match_hp_resolution


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
        # Fill any nans with median value
        to_add = np.where(np.isnan(to_add), np.nanmedian(to_add), to_add)
        # Resample the uncertainty
        to_add_uncert = match_hp_resolution(to_add_uncert, self.nside_out)
        # Probably should fill any missing values here too.

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

        # No frames loaded, return zero
        if np.size(self.mjds) == 0:
            if uncert:
                return 0, 0
            else:
                return 0

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

    def extinction_forecast(self, mjd, hpid=None, max_history=5, max_extrapolate_min=10.0):
        """Return a projected cloud extinction map for a future mjd.

        Tracks the recent motion of the cloud field by computing an
        extinction-weighted centroid (RA/Dec) of each of the last few
        stored frames, fitting a linear velocity to that centroid
        track, and advecting the most recent frame backward along
        that velocity to build a forecast map for the requested time.
        This is a simple translation (no growth/decay) model.

        Parameters
        ----------
        mjd : `float`
            MJD to forecast the extinction map for. Must be after the
            most recently added frame.
        hpid : `int`
            If only a subset of the HEALpix array is needed, can
            specify index.
        max_history : `int`
            Number of trailing frames to use for the velocity fit.
            Default 5.
        max_extrapolate_min : `float`
            Do not extrapolate more than this many minutes past the
            most recent frame; beyond that the linear-motion
            assumption is unreliable and we fall back to
            `extinction_closest`. Default 10.

        Returns
        -------
        Healpix array in ring order with projected cloud extinction
        in magnitudes (falls back to `extinction_closest` if there
        is not enough history, or the forecast fails, or the
        requested mjd is not far enough in the future).
        """
        pred = self._forecast_map(mjd, max_history=max_history, max_extrapolate_min=max_extrapolate_min)
        if pred is None:
            return self.extinction_closest(mjd, hpid=hpid)
        if hpid is not None:
            return pred[hpid]
        return pred

    def _forecast_map(self, target_mjd, max_history=5, max_extrapolate_min=10.0):
        """Build the advected forecast map, or None if not possible.

        Kept separate from `extinction_forecast` so the fallback to
        `extinction_closest` lives in one, easy-to-read place.
        """
        if len(self.mjds) < 2:
            return None
        use_n = min(max_history, len(self.mjds))
        recent_mjds = np.array(self.mjds[-use_n:])
        recent_maps = self.cloud_extinction_hparrays[-use_n:]

        lead_days = target_mjd - recent_mjds[-1]
        if lead_days <= 0 or lead_days * 1440.0 > max_extrapolate_min:
            return None

        npix = len(recent_maps[-1])
        nside = hp.npix2nside(npix)
        theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)
        dec = np.pi / 2 - theta

        centroids = []
        for g in recent_maps:
            finite = np.isfinite(g)
            if finite.sum() == 0:
                centroids.append(None)
                continue
            # Weight toward extinction (track the cloud, not the clear
            # sky), with a fixed/bounded observation footprint,
            # weighting toward low extinction tracks the clear-sky
            # centroid, which moves opposite the actual cloud motion
            # as the cloud sweeps across a bounded footprint.
            w = np.zeros_like(g)
            w[finite] = np.clip(g[finite], 0, None)
            total = w.sum()
            if total < 1e-9:
                centroids.append(None)
                continue
            # Circular mean for RA, not a plain average; RA wraps at
            # 2*pi, and the tracked footprint straddles that wrap on
            # most nights.
            sin_sum = (w * np.sin(phi)).sum()
            cos_sum = (w * np.cos(phi)).sum()
            ra_c = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)
            centroids.append((ra_c, float((w * dec).sum() / total)))

        valid = [(t, c) for t, c in zip(recent_mjds, centroids) if c is not None]
        if len(valid) < 2:
            return None
        ts = np.array([v[0] for v in valid])
        # np.unwrap removes the same 2*pi-wrap discontinuity from the
        # RA time series
        ras = np.unwrap(np.array([v[1][0] for v in valid]))
        decs = np.array([v[1][1] for v in valid])
        try:
            v_ra = np.polyfit(ts, ras, 1)[0]
            v_dec = np.polyfit(ts, decs, 1)[0]
        except Exception:
            return None

        d_ra = v_ra * lead_days
        d_dec = v_dec * lead_days
        # Advect the sampling grid backward by the fitted motion, then
        # look up each shifted position in the most recent frame; this
        # moves the frame's content forward by (d_ra, d_dec).
        src_dec = np.clip(dec - d_dec, -np.pi / 2, np.pi / 2)
        src_ra = phi - d_ra
        src_theta = np.pi / 2 - src_dec
        # Nearest-neighbor remap. hp.get_interp_val (bilinear) requires
        # all 4 neighbors finite, which starves real, sparse cloud
        # coverage down to a small fraction of valid pixels; nearest
        # neighbor is robust to gappy input.
        src_pix = hp.ang2pix(nside, src_theta, src_ra % (2 * np.pi))
        return recent_maps[-1][src_pix]
