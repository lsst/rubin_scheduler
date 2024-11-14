__all__ = ("DEFAULT_NSIDE", "SURVEY_START_MJD", "survey_start_mjd")

import warnings

DEFAULT_NSIDE = 32  # HEALpix nside, ~1.83 degree resolution
SURVEY_START_MJD = 60949.0  # 60949 = Oct 1, 2025


def survey_start_mjd():
    """For backwards compatibility"""
    warnings.warn("Function survey_start_mjd is deprecated, use SURVEY_START_MJD.", FutureWarning)

    return SURVEY_START_MJD
