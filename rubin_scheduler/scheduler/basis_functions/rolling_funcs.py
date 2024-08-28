__all__ = (
    "TargetMapModuloBasisFunction",
    "FootprintBasisFunction",
)

import warnings

import healpy as hp
import numpy as np

from rubin_scheduler.scheduler import features, utils
from rubin_scheduler.scheduler.basis_functions import BaseBasisFunction
from rubin_scheduler.scheduler.utils import ConstantFootprint


def send_unused_deprecation_warning(name):
    message = (
        f"The basis function {name} is not in use by the current "
        "baseline scheduler and may be deprecated shortly. "
        "Please contact the rubin_scheduler maintainers if "
        "this is in use elsewhere."
    )
    warnings.warn(message, FutureWarning)


class FootprintBasisFunction(BaseBasisFunction):
    """Basis function that tries to maintain a uniformly covered footprint

    Parameters
    ----------
    filtername : `str`, optional
        The filter for this footprint. Default r.
    nside : `int`, optional
        The nside for the basis function and features.
        Default None uses `~utils.set_default_nside()`
    footprint : `~rubin_scheduler.scheduler.utils.Footprint` object
        The desired footprint. The default will set this to None,
        but in general this is really not desirable.
        In order to make default a kwarg, a current baseline footprint
        is setup with a Constant footprint (not rolling, not even season
        aware).
    out_of_bounds_val : `float`, optional
        The value to set the basis function for regions that are not in
        the footprint. Default -10, np.nan is another good value to use.

    """

    def __init__(
        self,
        filtername="r",
        nside=None,
        footprint=None,
        out_of_bounds_val=-10.0,
    ):
        super().__init__(nside=nside, filtername=filtername)
        if footprint is None:
            # This is useful as a backup, but really footprint SHOULD
            # be specified when basis function is set up.
            # This just uses whole survey, but doesn't set up rolling.
            warnings.warn("No Footprint set, using a constant default.")
            target_maps, labels = utils.get_current_footprint(self.nside)
            fp = ConstantFootprint(nside=self.nside)
            for f in "ugrizy":
                fp.set_footprint(f, target_maps[f])
        self.footprint = footprint

        self.survey_features = {}
        # All the observations in all filters
        self.survey_features["N_obs_all"] = features.NObservations(nside=self.nside, filtername=None)
        self.survey_features["N_obs"] = features.NObservations(nside=self.nside, filtername=filtername)

        # should probably actually loop over all the target maps?
        self.out_of_bounds_area = np.where(footprint.get_footprint(self.filtername) <= 0)[0]
        self.out_of_bounds_val = out_of_bounds_val

    def _calc_value(self, conditions, indx=None):
        # Find out what the footprint object thinks we should have been
        # observed
        desired_footprint_normed = self.footprint(conditions.mjd)[self.filtername]

        # Compute how many observations we should have on the sky
        desired = desired_footprint_normed * np.sum(self.survey_features["N_obs_all"].feature)
        result = desired - self.survey_features["N_obs"].feature
        result[self.out_of_bounds_area] = self.out_of_bounds_val
        return result


class TargetMapModuloBasisFunction(BaseBasisFunction):
    """Basis function that tracks number of observations and tries to match
    a specified spatial distribution can enter multiple maps that will be
    used at different times in the survey

    Parameters
    ----------
    day_offset : np.array
        Healpix map that has the offset to be applied to each pixel when
        computing what season it is on.
    filtername : (string 'r')
        The name of the filter for this target map.
    nside: int (default_nside)
        The healpix resolution.
    target_maps : list of numpy array (None)
        healpix maps showing the ratio of observations desired for all
        points on the sky. Last map will be used for season -1. Probably
        shouldn't support going to season less than -1.
    norm_factor : float (0.00010519)
        for converting target map to number of observations. Should be the
        area of the camera divided by the area of a healpixel divided by
        the sum of all your goal maps. Default value assumes LSST foV has
        1.75 degree radius and the standard goal maps. If using mulitple
        filters, see rubin_scheduler.scheduler.utils.calc_norm_factor for
        a utility that computes norm_factor.
    out_of_bounds_val : float (-10.)
        Reward value to give regions where there are no observations
        requested (unitless).
    season_modulo : int (2)
        The value to modulate the season by (years).
    max_season : int (None)
        For seasons higher than this value (pre-modulo), the final target
        map is used.

    """

    def __init__(
        self,
        day_offset=None,
        filtername="r",
        nside=None,
        target_maps=None,
        norm_factor=None,
        out_of_bounds_val=-10.0,
        season_modulo=2,
        max_season=None,
        season_length=365.25,
    ):
        super(TargetMapModuloBasisFunction, self).__init__(nside=nside, filtername=filtername)

        if norm_factor is None:
            warnings.warn("No norm_factor set, use utils.calc_norm_factor if using multiple filters.")
            self.norm_factor = 0.00010519
        else:
            self.norm_factor = norm_factor

        self.survey_features = {}
        # Map of the number of observations in filter

        for i, temp in enumerate(target_maps[0:-1]):
            self.survey_features["N_obs_%i" % i] = features.N_observations_season(
                i,
                filtername=filtername,
                nside=self.nside,
                modulo=season_modulo,
                offset=day_offset,
                max_season=max_season,
                season_length=season_length,
            )
            # Count of all the observations taken in a season
            self.survey_features["N_obs_count_all_%i" % i] = features.N_obs_count_season(
                i,
                filtername=None,
                season_modulo=season_modulo,
                offset=day_offset,
                nside=self.nside,
                max_season=max_season,
                season_length=season_length,
            )
        # Set the final one to be -1
        self.survey_features["N_obs_%i" % -1] = features.N_observations_season(
            -1,
            filtername=filtername,
            nside=self.nside,
            modulo=season_modulo,
            offset=day_offset,
            max_season=max_season,
            season_length=season_length,
        )
        self.survey_features["N_obs_count_all_%i" % -1] = features.N_obs_count_season(
            -1,
            filtername=None,
            season_modulo=season_modulo,
            offset=day_offset,
            nside=self.nside,
            max_season=max_season,
            season_length=season_length,
        )
        if target_maps is None:
            target_maps, labels = utils.get_current_footprint(nside)
            self.target_map = target_maps[filtername]
        else:
            self.target_maps = target_maps
        # should probably actually loop over all the target maps?
        self.out_of_bounds_area = np.where(self.target_maps[0] == 0)[0]
        self.out_of_bounds_val = out_of_bounds_val
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.all_indx = np.arange(self.result.size)

        # For computing what day each healpix is on
        if day_offset is None:
            self.day_offset = np.zeros(hp.nside2npix(self.nside), dtype=float)
        else:
            self.day_offset = day_offset

        self.season_modulo = season_modulo
        self.max_season = max_season
        self.season_length = season_length
        send_unused_deprecation_warning(self.__class__.__name__)

    def _calc_value(self, conditions, indx=None):
        """
        Parameters
        ----------
        indx : list (None)
            Index values to compute, if None, full map is computed
        Returns
        -------
        Healpix reward map
        """

        result = self.result.copy()
        if indx is None:
            indx = self.all_indx

        # Compute what season it is at each pixel
        seasons = utils.season_calc(
            conditions.night,
            offset=self.day_offset,
            modulo=self.season_modulo,
            max_season=self.max_season,
            season_length=self.season_length,
        )

        composite_target = self.result.copy()[indx]
        composite_nobs = self.result.copy()[indx]

        composite_goal_n = self.result.copy()[indx]

        for season in np.unique(seasons):
            season_indx = np.where(seasons == season)[0]
            composite_target[season_indx] = self.target_maps[season][season_indx]
            composite_nobs[season_indx] = self.survey_features["N_obs_%i" % season].feature[season_indx]
            composite_goal_n[season_indx] = (
                composite_target[season_indx]
                * self.survey_features["N_obs_count_all_%i" % season].feature
                * self.norm_factor
            )

        result[indx] = composite_goal_n - composite_nobs[indx]
        result[self.out_of_bounds_area] = self.out_of_bounds_val

        return result
