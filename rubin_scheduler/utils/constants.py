__all__ = ("DEFAULT_NSIDE", "SURVEY_START_MJD", "survey_start_mjd", "JD2MJDOFFSET")

import warnings

DEFAULT_NSIDE = 32  # HEALpix nside, ~1.83 degree resolution
SURVEY_START_MJD = 60980.0  # 60980 = Nov 1, 2025
JD2MJDOFFSET = 2400000.5  # MJD = JD - 2400000.5


def survey_start_mjd():
    """For backwards compatibility"""
    warnings.warn("Function survey_start_mjd is deprecated, use SURVEY_START_MJD.", FutureWarning)

    return SURVEY_START_MJD
