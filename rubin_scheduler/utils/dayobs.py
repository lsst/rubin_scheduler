import numpy as np
from astropy.time import Time


def mjd2dayobs(mjd, offset=-0.5):
    """Convert MJD to Rubin DayObs.

    Parameters
    ----------
    mjd : `float`
        Modified Julian Date.
    offset : `float`
        Offset to apply to the MJD. For Rubin, should be
        -0.5 (days).

    Returns
    -------
    dayobs_str : `string`
        The dayobs as a string, yyyymmdd.
    """
    atime = Time(mjd + offset, format="mjd")
    ymd = atime.to_value("ymdhms")
    year_str = np.char.zfill(ymd["year"].astype(str), 4)
    mo_str = np.char.zfill(ymd["month"].astype(str), 2)
    day_str = np.char.zfill(ymd["day"].astype(str), 2)
    dayobs_str = np.char.add(year_str, mo_str)
    dayobs_str = np.char.add(dayobs_str, day_str)
    return dayobs_str
