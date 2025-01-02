__all__ = ("VaryExptDetailer", "calc_target_m5s")

import healpy as hp
import numpy as np
from scipy.stats import binned_statistic

from rubin_scheduler.scheduler.detailers import BaseDetailer
from rubin_scheduler.skybrightness_pre import dark_sky
from rubin_scheduler.utils import DEFAULT_NSIDE, Site, _ra_dec2_hpid, hpid2_ra_dec, m5_flat_sed


def calc_target_m5s(alt=65.0, fiducial_seeing=0.9, exptime=20.0):
    """Use the skybrightness model to find some good target m5s.

    Parameters
    ----------
    alt : `float`, opt
        Altitude for the target, degrees. Default 65.
    fiducial_seeing : `float`, opt
        Fiducial FWHMeff seeing, arcseconds. Default 0.9.
    exptime : `float`, opt
        Exposure time for the comparison, seconds. Default 20.

    Returns
    -------
    goal_m5 : `dict` of `float`
        dictionary of expected m5 values keyed by bandname
    """

    nside = DEFAULT_NSIDE
    dark = dark_sky(nside=nside)
    hpid = np.arange(dark.size, dtype=int)
    ra, dec = hpid2_ra_dec(nside, hpid)
    site = Site(name="LSST")
    alts = site.latitude - dec + 90
    alts[np.where(alts > 90)] -= 90
    binsize = 5.0
    alt_bins = np.arange(0, 90 + binsize, binsize)
    alts_mid = (alt_bins[0:-1] + alt_bins[1:]) / 2
    sky_mags = {}
    high_alts = np.where(alts > 0)[0]
    for bandname in dark.dtype.names:
        sky_mags[bandname], _be, _binn = binned_statistic(
            alts[high_alts], dark[bandname][high_alts], bins=alt_bins, statistic="mean"
        )
        sky_mags[bandname] = np.interp(alt, alts_mid, sky_mags[bandname])

    airmass = 1.0 / np.cos(np.pi / 2.0 - np.radians(alt))

    goal_m5 = {}
    for bandname in sky_mags:
        goal_m5[bandname] = m5_flat_sed(bandname, sky_mags[bandname], fiducial_seeing, exptime, airmass)

    return goal_m5


class VaryExptDetailer(BaseDetailer):
    """Vary the exposure time on observations to try and keep each
    observation at uniform depth.

    Parameters
    ----------
    min_expt : `float` (20.)
        The minimum exposure time to use (seconds).
    max_expt : `float` (100.)
        The maximum exposure time to use
    target_m5 : `dict` (None)
        Dictionary with keys of bandnames as str and target 5-sigma
        depth values as floats. If none, the target_m5s are set to a
        min_expt exposure at X=1.1 in dark time.

    """

    def __init__(self, nside=DEFAULT_NSIDE, min_expt=20.0, max_expt=100.0, target_m5=None):
        """"""
        # Dict to hold all the features we want to track
        self.survey_features = {}
        self.nside = nside
        self.min_exp = min_expt
        self.max_exp = max_expt
        if target_m5 is None:
            self.target_m5 = {
                "g": 24.381615425253738,
                "i": 23.41810142458083,
                "r": 23.964359143049755,
                "u": 22.978794343692783,
                "y": 21.755612950787068,
                "z": 22.80377793629767,
            }
        else:
            self.target_m5 = target_m5

    def __call__(self, obs_array, conditions):
        """
        Parameters
        ----------
        observation_list : `list` of observations
            The observations to detail.
        conditions : `rubin_scheduler.scheduler.conditions` object

        Returns
        -------
        List of observations.
        """
        hpids = _ra_dec2_hpid(self.nside, obs_array["RA"], obs_array["dec"])
        new_expts = np.zeros(obs_array.size, dtype=float)
        for bandname in np.unique(obs_array["band"]):
            in_filt = np.where(obs_array["band"] == bandname)
            delta_m5 = self.target_m5[bandname] - conditions.m5_depth[bandname][hpids[in_filt]]
            # We can get NaNs because dithering pushes the center of the
            # pointing into masked regions.
            nan_indices = np.argwhere(np.isnan(delta_m5)).ravel()
            for indx in nan_indices:
                bad_hp = hpids[in_filt][indx]
                # Note this might fail if we run at higher resolution,
                # then we'd need to look farther for pixels to interpolate.
                near_pix = hp.get_all_neighbours(conditions.nside, bad_hp)
                vals = conditions.m5_depth[bandname][near_pix]
                if True in np.isfinite(vals):
                    estimate_m5 = np.mean(vals[np.isfinite(vals)])
                    delta_m5[indx] = self.target_m5[bandname] - estimate_m5
                else:
                    raise ValueError("Failed to find a nearby unmasked sky value.")

            new_expts[in_filt] = conditions.exptime * 10 ** (delta_m5 / 1.25)
        new_expts = np.clip(new_expts, self.min_exp, self.max_exp)
        # I'm not sure what level of precision we can expect, so let's
        # just limit to seconds
        obs_array["exptime"] = np.round(new_expts)

        return obs_array
