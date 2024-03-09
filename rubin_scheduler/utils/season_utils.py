__all__ = ("calc_season",)

import numpy as np
from astropy.coordinates import EarthLocation, get_sun
from astropy.time import Time

from .mjd_zero import survey_start_mjd


def calc_season(ra, mjd, mjd_start=None):
    """Calculate the season (of visibility) in the survey
    for a series of ra/time values of an observation.

    Based only on the RA of the point on the
    sky, it calculates the 'season' based on when the sun
    passes through this RA (this marks the start of a 'season').


    Parameters
    ----------
    ra : `float` or `np.ndarray` (N,)
        The RA (in degrees) of the point(s) on the sky
    mjd : `np.ndarray`, (M,)
        The times of the observations, in MJD days
    mjd_start : `float`, optional
        The time of the start of the survey.
        Default None will use `rubin_scheduler.utils.survey_start_mjd()`.

    Returns
    -------
    seasons : `np.array`, (M,) or `np.array` (N, M)
        The season values, as either a 1-d or 2-d array, depending on
        the value passed for ra.


    Notes
    -----
    Seasons should be calculated using the RA of a fixed point
    on the sky, such as the slice_point['ra'] if calculating season
    values for a series of opsim pointings on the sky, rather
    than the value for each visit.

    To convert to integer seasons, use np.floor(seasons).

    "Season 1" will be the first season after the RA value becomes
    visible for the first time after mjd_start.
    This induces a "skip" in the resulting season values, corresponding
    to the RA where the survey started.

    The scheduler computes "season" independently, and may have a 90 degree
    offset in the location of this 'skip'/zeropoint.
    """
    # Just to make it simpler to get simple values back
    try:
        len(ra)
        single_ra = False
    except TypeError:
        single_ra = True

    if mjd_start is None:
        mjd_start = survey_start_mjd()

    # A reference time and sun RA location to anchor the location of the Sun
    # This time was chosen as it is close to the expected start of the survey.
    # The time is an equinox point, so the sun is at dec=0.
    ref_time = 60940.0
    ref_sun_ra = 178.98403541421598

    # Calculate the fraction of the sphere/"year" for these RAs on the sky.
    offset = (ra - ref_sun_ra) / 360 * 365.25
    # Turn the offsets into the (mjd) times that the season would start
    # at each RA value, in the reference year.
    season_began = ref_time + offset

    # Add an adjustment so that the first season at each RA is in the year
    # after mjd_start
    first = np.floor((season_began - mjd_start) / 365.25)
    season_began = season_began - first * 365.25

    # Calculate the season value for each point.
    # season_began is an N length array, matching "ra"
    # mjds is an M length array, matching mjds (applied to every ra)
    if single_ra:
        seasons = (mjd - season_began) / 365.25
    else:
        seasons = np.zeros((len(ra), len(mjd)), dtype=float)
        for i, r in enumerate(ra):
            seasons[i] = (mjd - season_began[i]) / 365.25

    return seasons

def _generate_reference():
    # The reference values for calc_season can be evaluated using
    loc = EarthLocation.of_site("Rubin", refresh_cache=True)
    # time should ideally be near at equinox for RA projection considerations
    t = Time("2025-09-22T00:00:00.00", format="isot", scale="utc", location=loc)
    print("Ref time", t.utc.mjd)
    print("Ref sun RA", get_sun(t).ra.deg, t.utc.mjd)
    print("local sidereal time at time", t.sidereal_time("apparent").deg)