import warnings

import numpy as np

from rubin_scheduler.scheduler.detailers import BaseDetailer


class BandPickToODetailer(BaseDetailer):
    """Select which filters are used based on information in a
    ToO event.

    Parameters
    ----------
    band_start : `str`
        Which band to consider changing
    band_end : `str`
        What band to change to.
    distance_limit : `float`
        For distances less than distance_limit,
        the bands are swapped. (kpc)
    check_mounted : `bool`
        Check that the desired band is mounted before changing
        values.
    require_dark : `bool`
        If in dark conditions, the bands are swapped.
    """

    def __init__(
        self, band_start="i", band_end="z", distance_limit=10, check_mounted=True, require_dark=False
    ):
        self.band_start = band_start
        self.band_end = band_end
        self.distance_limit = distance_limit
        self.check_mounted = check_mounted
        self.require_dark = require_dark

    def __call__(self, observations, conditions, target_o_o=None):

        if hasattr(target_o_o, "posterior_distance"):
            if target_o_o.posterior_distance is not None:
                if self.require_dark and conditions.moon_phase > 50:
                    warnings.warn("Requires dark time, moon phase is greater than 50%.")
                elif target_o_o.posterior_distance < self.distance_limit:
                    if self.check_mounted:
                        mounted = self.band_end in conditions.mounted_bands
                    else:
                        mounted = True
                    if mounted:
                        band_start_indx = np.where(observations["band"] == self.band_start)[0]
                        observations["band"][band_start_indx] = self.band_end
            else:
                warnings.warn("No distance present, not swapping bands.")
        else:
            warnings.warn("No distance present, not swapping bands.")
        return observations
