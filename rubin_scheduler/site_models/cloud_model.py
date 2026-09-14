__all__ = ("CloudModel",)

import warnings

import healpy as hp
import numpy as np

from rubin_scheduler.utils import DEFAULT_NSIDE


class CloudModel:
    """LSST cloud calculations for predictive cloud extinction.

    Converts a short window of recent DREAM cloud-extinction frames
    into a delivered extinction map projected forward to a requested
    time.

    Parameters
    ----------
    nside_out : `int`
        The nside to output maps as. Default DEFAULT_NSIDE
        (probably 32).
    max_history : `int`
        Number of trailing frames (out of whatever is in
        `dream_history`) to use for the velocity fit. Default 5.
    max_extrapolate_min : `float`
        Do not extrapolate more than this many minutes past the most
        recent frame in `dream_history`; beyond that the
        linear-motion assumption is unreliable and we fall back to
        the most recently observed frame, unmodified. Default 10.

    self.efd_requirements and self.map_requirements follow the same
    convention as the other site_models: efd_requirements is a tuple
    (list of str, float) naming the DREAM/EFD columns and the amount
    of time history required; map_requirements is a list of str
    naming the columns required in the target map dict.
    """

    def __init__(self, nside_out=DEFAULT_NSIDE, max_history=5, max_extrapolate_min=10.0):
        self.nside_out = nside_out
        self.max_history = max_history
        self.max_extrapolate_min = max_extrapolate_min
        # Matches the SeeingModel/CloudModel(placeholder) convention:
        # what this model needs pulled from DREAM/EFD, and for how
        # long a history.
        self.efd_requirements = (["mjd", "extinction"], max_history)
        self.map_requirements = ["mjd"]

    def configure(self, config=None):
        """Deprecated. Configure through the init method."""
        warnings.warn("The configure method is deprecated.")

    def config_info(self):
        """Deprecated. Report configuration parameters and version
        information."""
        warnings.warn("The config_info method is deprecated.")

    def __call__(self, dream_history, mjd):
        """Deliver a projected cloud extinction map for `mjd`.

        Parameters
        ----------
        dream_history : `dict` or object with `mjds` /
            `cloud_extinction_hparrays` attributes (e.g. a
            `~.site_models.CloudMap`-like buffer works directly).
            A short recent window of DREAM frames:
            dream_history["mjd"] : `np.array`
                MJDs of the recent frames, oldest to newest.
            dream_history["extinction"] : `list` [`np.array`]
                HEALpix extinction maps (ring order, `nside_out`)
                for each of those MJDs, same order.
        mjd : `float`
            The time to deliver the extinction map for. Must be
            after the most recent frame in `dream_history`.

        Returns
        -------
        dict of `np.ndarray`
            {"extinction": healpix array in ring order, magnitudes}.
            Falls back to the most recently observed frame, unmodified,
            if there isn't enough history, the forecast fails, or
            `mjd` is not far enough in the future to be worth
            extrapolating (see `max_extrapolate_min`).
        """
        if not isinstance(mjd, (int, float, np.integer, np.floating)):
            return self._legacy_call(dream_history, mjd)

        if isinstance(dream_history, dict):
            recent_mjds = np.asarray(dream_history["mjd"])
            recent_maps = list(dream_history["extinction"])
        else:
            recent_mjds = np.asarray(dream_history.mjds)
            recent_maps = list(dream_history.cloud_extinction_hparrays)

        pred = self._forecast(recent_mjds, recent_maps, mjd)
        if pred is None:
            # Not enough history, or mjd not far enough ahead to be
            # worth extrapolating; deliver the most recent observed
            # frame unmodified (the old extinction_closest behavior).
            fallback = recent_maps[-1] if len(recent_maps) else np.zeros(hp.nside2npix(self.nside_out))
            return {"extinction": fallback}
        return {"extinction": pred}

    def _legacy_call(self, cloud_value, altitude):
        """Original placeholder behavior: broadcast a single cloud
        value over the sky."""
        warnings.warn(
            "CloudModel(cloud_value, altitude) is deprecated in favor of "
            "CloudModel(dream_history, mjd); see the __call__ docstring.",
            FutureWarning,
        )
        if isinstance(cloud_value, dict):
            cloud_value = cloud_value["cloud"]
        if isinstance(altitude, dict):
            altitude = altitude["altitude"]
        model_cloud = np.zeros(len(altitude), float) + cloud_value
        return {"cloud": model_cloud}

    def _forecast(self, recent_mjds, recent_maps, target_mjd):
        """Weighted-centroid + linear-velocity cloud advection.

        Same validated method as
        `~.site_models.CloudMap.extinction_forecast`: track the
        extinction-weighted centroid of each recent frame, fit a
        linear velocity to that centroid track, and advect the most
        recent frame along that velocity to build the forecast for
        `target_mjd`. Nearest-neighbor remap (not bilinear
        `hp.get_interp_val`, which needs all 4 neighbors finite and
        starves real, sparse cloud coverage).
        """
        if len(recent_mjds) < 2:
            return None
        use_n = min(self.max_history, len(recent_mjds))
        recent_mjds = recent_mjds[-use_n:]
        recent_maps = recent_maps[-use_n:]

        lead_days = target_mjd - recent_mjds[-1]
        if lead_days <= 0 or lead_days * 1440.0 > self.max_extrapolate_min:
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
            # sky)
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
        # np.unwrap removes the same 2*pi discontinuity from the RA
        # time series so a real, smooth drift across the wrap isn't
        # seen by polyfit as a huge spurious jump.
        ras = np.unwrap(np.array([v[1][0] for v in valid]))
        decs = np.array([v[1][1] for v in valid])
        try:
            v_ra = np.polyfit(ts, ras, 1)[0]
            v_dec = np.polyfit(ts, decs, 1)[0]
        except Exception:
            return None

        d_ra = v_ra * lead_days
        d_dec = v_dec * lead_days
        src_dec = np.clip(dec - d_dec, -np.pi / 2, np.pi / 2)
        src_ra = phi - d_ra
        src_theta = np.pi / 2 - src_dec
        src_pix = hp.ang2pix(nside, src_theta, src_ra % (2 * np.pi))
        return recent_maps[-1][src_pix]
