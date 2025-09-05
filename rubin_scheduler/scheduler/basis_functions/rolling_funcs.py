__all__ = ("FootprintBasisFunction",)

import warnings

import numpy as np

from rubin_scheduler.scheduler import features, utils
from rubin_scheduler.scheduler.basis_functions import BaseBasisFunction
from rubin_scheduler.scheduler.utils import ConstantFootprint
from rubin_scheduler.utils import DEFAULT_NSIDE


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
    bandname : `str`, optional
        The band for this footprint. Default r.
    nside : `int`, optional
        The nside for the basis function and features.
        Default None uses `~utils.set_default_nside()`
    footprint : `~rubin_scheduler.scheduler.utils.Footprint` object
        The desired footprint. The default will set this to None,
        but in general this is really not desirable.
        In order to make default a kwarg, a current baseline footprint
        is set up with a Constant footprint (not rolling, not even season
        aware).
    out_of_bounds_val : `float`, optional
        The value to set the basis function for regions that are not in
        the footprint. Default -10, np.nan is another good value to use.
    seeing_fwhm_max : `float`
        Seeing limit to use (arcsec) when counting observations.
        Default None.
    """

    def __init__(
        self,
        bandname="r",
        nside=DEFAULT_NSIDE,
        footprint=None,
        out_of_bounds_val=-10.0,
        filtername=None,
        seeing_fwhm_max=None,
    ):
        if filtername is not None:
            warnings.warn("filtername deprecated in favor of bandname", FutureWarning)
            bandname = filtername
        super().__init__(nside=nside, bandname=bandname)
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
        # All the observations in all bands
        self.survey_features["N_obs_all"] = features.NObservations(
            nside=self.nside, bandname=None, seeing_limit=seeing_fwhm_max
        )
        self.survey_features["N_obs"] = features.NObservations(
            nside=self.nside, bandname=bandname, seeing_limit=seeing_fwhm_max
        )

        # should probably actually loop over all the target maps?
        self.out_of_bounds_area = np.where(footprint.get_footprint(self.bandname) <= 0)[0]
        self.out_of_bounds_val = out_of_bounds_val

    def _calc_value(self, conditions, indx=None):
        # Find out what the footprint object thinks we should have been
        # observed
        desired_footprint_normed = self.footprint(conditions.mjd)[self.bandname]

        # Compute how many observations we should have on the sky
        desired = desired_footprint_normed * np.sum(self.survey_features["N_obs_all"].feature)
        result = desired - self.survey_features["N_obs"].feature
        result[self.out_of_bounds_area] = self.out_of_bounds_val
        return result
