__all__ = ("calc_season",)

import numpy as np
from astropy.coordinates import EarthLocation, get_sun
from astropy.time import Time


def calc_season(ra, mjd):
    """Calculate the 'season' in the survey for a series of ra/time
    values of an observation. Based only on the RA of the point on the
    sky, it calculates the 'season' based on when the sun passes through
    this RA (this marks the start of a 'season').

    Note that seasons should be calculated using the RA of a fixed point
    on the sky, such as the slice_point['ra'] if calculating season
    values for a series of opsim pointings on the sky. To convert to
    integer seasons, use np.floor(seasons)

    Parameters
    ----------
    ra : `float`
        The RA (in degrees) of the point on the sky
    mjd : `np.ndarray`
        The times of the observations, in MJD days
    mjd_start : `float`
        The MJD for the start of the survey. If None, uses
        minimum of input mjd to set a zeropoint
    ref_RA : `float`
        The reference RA to use when setting the season
        zeropoint. If None, uses minimum of input RA.

    Returns
    -------
    seasons : `np.array`
        The season values, as floats.
    """

    if np.size(ra) > 1:
        ValueError("The ra must be a single value, not array.")

    # A reference time and sun RA location to anchor the location of the Sun
    # This time was chosen as it is close to the expected start of the survey.
    ref_time = 60575.0
    ref_sun_ra = 179.20796047239727
    # Calculate the fraction of the sphere/"year" for this location
    offset = (ra - ref_sun_ra) / 360 * 365.25
    # Calculate when the seasons should begin
    season_began = ref_time + offset
    # Calculate the season value for each point.
    seasons = (mjd - season_began) / 365.25

    seasons = seasons - np.floor(np.min(seasons))
    return seasons


def _generate_reference():
    # The reference values for calc_season can be evaluated using
    loc = EarthLocation.of_site("Rubin")
    t = Time("2024-09-22T00:00:00.00", format="isot", scale="utc", location=loc)
    print("Ref time", t.utc.mjd)
    print("Ref sun RA", get_sun(t).ra.deg, t.utc.mjd)
    print("local sidereal time at season start", t.sidereal_time("apparent").deg)
