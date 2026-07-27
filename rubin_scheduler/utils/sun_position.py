__all__ = ("NextTimeSun",)

import numpy as np

from astropy.coordinates import get_sun, AltAz, EarthLocation

from astropy.time import Time
from rubin_scheduler.utils import Site
from scipy.optimize import minimize


class NextTimeSun(object):
    """Find the next time the sun will be at some altitude.
    Could be done with astroplan, but trying to save a dependency.

    Note: Probably fails at extreem latitudes when the sun doesn't
    rise/set for many days at a time.

    Parameter
    ---------
    location : `astropy.coordinates.EarthLocation`
        Location of the observatory. Defaults to LSST.
    """

    def __init__(self, location=None):
        if location is None:
            site = Site("LSST")
            self.location = EarthLocation(lat=site.latitude, lon=site.longitude, height=site.height)
        else:
            self.location = location
        self.frame = AltAz(location=self.location)

    def _call_to_min(self, mjd):
        """Method that can be used by minimization routine."""
        return (self.alt_at_mjd(mjd) - self.altitude) ** 2

    def alt_at_mjd(self, mjd):
        """Return sun altitude in degrees for given MJD."""
        sun_altaz = get_sun(Time(mjd, format="mjd")).transform_to(self.frame)
        return sun_altaz.alt.deg

    def next_mjd_at_alt(
        self, mjd, altitude=-12.0, rising=True, time_steps=20, forward_check_length=1.5, **kwargs
    ):
        """Find the time the sun will next be at a given altitude.

        Parameters
        ----------
        mjd : `float`
            The modified Julian Date.
        altitude : `float`
            Altitude for the sun (Degrees). Default -12.
        rising : `bool`
            Should the sun be rising (True) or setting (False).
            Default True.
        time_steps : `int`
            How many time steps to use when finding next time.
            Default 20.
        forward_check_length : `float`
            How far into the future to look for the next sun
            positions. Default 1.5 (days)
        **kwargs
            Passed to scipy.optimize.minimize.
        """
        self.altitude = altitude
        tsteps = np.linspace(0, forward_check_length, num=time_steps)
        times = Time(mjd + tsteps, format="mjd")
        sun_altaz = get_sun(times).transform_to(self.frame)
        # Get the slopes
        diff_limit = sun_altaz.alt.deg - altitude
        # If there is an amazing lucky strike and
        # we hit the exact floating point precicion time
        if 0 in diff_limit:
            return times[np.where(diff_limit == 0)].mjd
        rise_set_sign = np.diff(np.sign(diff_limit))
        # Sun setting when ack == -2, rising when ack ==2
        if rising:
            indx = np.min(np.where(rise_set_sign == 2)[0])
        else:
            indx = np.min(np.where(rise_set_sign == -2)[0])
        x0 = times[indx].mjd
        bounds = [(times[indx - 1].mjd, times[indx + 1].mjd)]
        result = minimize(self._call_to_min, x0, bounds=bounds, **kwargs)

        # take a max so we are sure to return a scalar.
        return np.max(result.x)
