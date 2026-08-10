__all__ = ("BrightObservatoryModel",)

import numpy as np
from astropy.time import Time

from rubin_scheduler.utils import SURVEY_START_MJD, NextTimeSun, _angular_separation

from .model_observatory import ModelObservatory


class BrightObservatoryModel(ModelObservatory):
    """Observatory that can go beyond sun altitude of -12 deg.

    Parameters
    ----------
    sun_rise_limit_deg : `float`
        The sun altitude limit where we should skip ahead
        to the next sunset. Default -9 (degrees).
    sun_set_limit_deg : `float`
        What should the altitude of the sun be at sunset
        when observing starts. Default -9 (degrees).
    delta_time_step_sec : `float`
        A fudge time to make sure the sun is really higher
        than sun_set_limit_deg. Default 0.1 (seconds).
    **kwargs
        The usual ModelObservatory kwargs that get passed
        along.
    """

    def __init__(
        self,
        mjd_start=SURVEY_START_MJD,
        mjd=None,
        sun_rise_limit_deg=-9,
        sun_set_limit_deg=-9,
        delta_time_step_sec=0.1,
        **kwargs,
    ):

        if mjd is None:
            mjd = mjd_start

        super().__init__(**kwargs)

        self.sun_rise_limit_deg = sun_rise_limit_deg
        self.sun_set_limit_deg = sun_set_limit_deg
        self.delta_time_step = delta_time_step_sec / 3600 / 24  # to days
        self.sun_moon_lookup = NextTimeSun(location=self.location)
        self.set_initial_mjd(mjd)

    def add_sun_moon_data(self, observation):
        """Add info about the sun and moon. Not using interpolated
        values from the almanac.
        """
        sun, sun_frame, moon, moon_frame = self.sun_moon_lookup.sun_moon(observation["mjd"])

        observation["sunAlt"] = sun_frame.alt.rad
        observation["sunAz"] = sun_frame.az.rad
        observation["sunRA"] = sun.ra.rad
        observation["sunDec"] = sun.dec.rad
        observation["moonAlt"] = moon_frame.alt.rad
        observation["moonAz"] = moon_frame.az.rad
        observation["moonRA"] = moon.ra.rad
        observation["moonDec"] = moon.dec.rad
        observation["moonDist"] = _angular_separation(
            observation["RA"],
            observation["dec"],
            observation["moonRA"],
            observation["moonDec"],
        )
        observation["solarElong"] = _angular_separation(
            observation["RA"],
            observation["dec"],
            observation["sunRA"],
            observation["sunDec"],
        )

        # Moon phase from
        # https://github.com/egemenimre/satstuff/blob/
        # master/notebooks/astropy/moon_venus_phase.ipynb
        sun_vec = sun.cartesian
        moon_vec = moon.cartesian

        gnd_loc = self.location.get_gcrs(
            Time(observation["mjd"], format="mjd")
        ).cartesian.without_differentials()

        # Generate Sun, Moon-to-location vectors
        sun_to_moon = sun_vec - moon_vec
        gnd_to_moon = gnd_loc - moon_vec

        sun_to_moon_unit = sun_to_moon / sun_to_moon.norm()
        gnd_to_moon_unit = gnd_to_moon / gnd_to_moon.norm()

        phase_angle_moon = np.arccos(sun_to_moon_unit.dot(gnd_to_moon_unit))
        observation["moonPhase"] = phase_angle_moon / np.pi * 100

        return observation

    def check_mjd(self, mjd, cloud_skip=20.0):
        """See if an mjd is ok to observe

        Parameters
        ----------
        cloud_skip : float (20)
            How much time to skip ahead if it's cloudy (minutes)

        Returns
        -------
        mjd_ok : `bool`
        mdj : `float`
            If True, the input mjd. If false, a good mjd to skip
            forward to.
        """
        passed = True
        new_mjd = mjd + 0

        clouds = self.cloud_data(Time(mjd, format="mjd"))

        if clouds > self.cloud_limit:
            passed = False
            while clouds > self.cloud_limit:
                new_mjd = new_mjd + cloud_skip / 60.0 / 24.0
                clouds = self.cloud_data(Time(new_mjd, format="mjd"))
        # at the end of the night, advance to the next setting twilight
        sun_alt = self.sun_moon_lookup.alt_at_mjd(new_mjd)
        if sun_alt > self.sun_rise_limit_deg:
            passed = False
            new_mjd = self.sun_moon_lookup.next_mjd_at_alt(
                new_mjd, altitude=self.sun_set_limit_deg, rising=False
            )
            # Add a fudge since the new_mjd is from a fit that can be
            # off by machine precision
            new_mjd += self.delta_time_step

        # We're in a down time, if down, advance to the end of the downtime
        if not self.check_up(mjd)[0]:
            passed = False
            new_mjd = self.check_up(mjd)[1]
        # recursive call to make sure we skip far enough ahead
        if not passed:
            while not passed:
                passed, new_mjd = self.check_mjd(new_mjd)
            return False, new_mjd
        else:
            return True, mjd
