__all__ = ("DEFAULT_NSIDE", "SURVEY_START_MJD", "survey_start_mjd")

import warnings

from astropy.time import Time

DEFAULT_NSIDE = 32  # HEALpix nside, ~1.83 degree resolution
SURVEY_START_MJD = Time("2025-11-01T12:00:00").mjd  # Should be during the day before first observation


def survey_start_mjd():
    """For backwards compatibility"""
    warnings.warn("Function survey_start_mjd is deprecated, use SURVEY_START_MJD.", FutureWarning)

    return SURVEY_START_MJD
