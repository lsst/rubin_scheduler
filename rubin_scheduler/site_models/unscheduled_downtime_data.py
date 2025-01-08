__all__ = ("UnscheduledDowntimeData", "UnscheduledDowntimeMoreY1Data")

import random

import numpy as np
from astropy.time import Time, TimeDelta

from . import Almanac


class UnscheduledDowntimeData:
    """Handle (and create) the unscheduled downtime information.

    Parameters
    ----------
    start_time : `astropy.time.Time`
        The time of the start of the simulation.
        The cloud database will be assumed to start on Jan 01 of the
        same year.
    seed : `int`, optional
        The random seed for creating the random nights of unscheduled
        downtime. Default 1516231120.
    start_of_night_offset : `float`, optional
        The fraction of a day to offset from MJD.0 to reach the
        defined start of a night ('noon' works).
        Default 0.16 (UTC midnight in Chile) - 0.5
        (minus half a day) = -0.34
    survey_length : `int`, optional
        The number of nights in the total survey. Default 3650*2.
    """

    MINOR_EVENT = {"P": 0.0137, "length": 1, "level": "minor event"}
    INTERMEDIATE_EVENT = {"P": 0.00548, "length": 3, "level": "intermediate event"}
    MAJOR_EVENT = {"P": 0.00137, "length": 7, "level": "major event"}
    CATASTROPHIC_EVENT = {"P": 0.000274, "length": 14, "level": "catastrophic event"}

    def __init__(
        self,
        start_time,
        seed=1516231120,
        start_of_night_offset=-0.34,
        survey_length=3650 * 2,
    ):
        self.seed = seed
        self.survey_length = survey_length
        self.start_of_night_offset = start_of_night_offset
        year_start = start_time.datetime.year
        self.night0 = Time("%d-01-01" % year_start, format="isot", scale="tai") + TimeDelta(
            start_of_night_offset, format="jd"
        )
        self.start_time = start_time

        # Scheduled downtime data is a np.ndarray of start
        # / end / activity for each scheduled downtime.
        self.downtime = None
        self.make_data()

    def __call__(self):
        """Return the array of unscheduled downtimes.

        Parameters
        ----------
        time : `astropy.time.Time`
            Time in the simulation for which to find the current downtime.

        Returns
        -------
        downtime : `np.ndarray`
            The array of all unscheduled downtimes, with keys for
            'start', 'end', 'activity',  corresponding to
            `astropy.time.Time`, `astropy.time.Time`, and `str`.
        """
        return self.downtime

    def _downtime_status(self, time):
        """Look behind the scenes at the downtime status/next values"""
        next_start = self.downtime["start"].searchsorted(time, side="right")
        next_end = self.downtime["end"].searchsorted(time, side="right")
        if next_start > next_end:
            current = self.downtime[next_end]
        else:
            current = None
        future = self.downtime[next_start:]
        return current, future

    def make_data(self):
        """Configure the set of unscheduled downtimes.

        This function creates the unscheduled downtimes based on a set
        of probabilities of the downtime type occurance.

        The random downtime is calculated using the following
        probabilities:

        minor event : remainder of night and next day = 5/365 days
        e.g. power supply failure
        intermediate : 3 nights = 2/365 days e.g. repair band
        mechanism, rotator, hexapod, or shutter
        major event : 7 nights = 1/2*365 days
        catastrophic event : 14 nights = 1/3650 days e.g. replace a raft
        """
        random.seed(self.seed)

        starts = []
        ends = []
        acts = []
        night = 0
        while night < self.survey_length:
            prob = random.random()
            if prob < self.CATASTROPHIC_EVENT["P"]:
                start_night = self.night0 + TimeDelta(night, format="jd")
                starts.append(start_night)
                end_night = start_night + TimeDelta(self.CATASTROPHIC_EVENT["length"], format="jd")
                ends.append(end_night)
                acts.append(self.CATASTROPHIC_EVENT["level"])
                night += self.CATASTROPHIC_EVENT["length"] + 1
                continue
            else:
                prob = random.random()
                if prob < self.MAJOR_EVENT["P"]:
                    start_night = self.night0 + TimeDelta(night, format="jd")
                    starts.append(start_night)
                    end_night = start_night + TimeDelta(self.MAJOR_EVENT["length"], format="jd")
                    ends.append(end_night)
                    acts.append(self.MAJOR_EVENT["level"])
                    night += self.MAJOR_EVENT["length"] + 1
                    continue
                else:
                    prob = random.random()
                    if prob < self.INTERMEDIATE_EVENT["P"]:
                        start_night = self.night0 + TimeDelta(night, format="jd")
                        starts.append(start_night)
                        end_night = start_night + TimeDelta(self.INTERMEDIATE_EVENT["length"], format="jd")
                        ends.append(end_night)
                        acts.append(self.INTERMEDIATE_EVENT["level"])
                        night += self.INTERMEDIATE_EVENT["length"] + 1
                        continue
                    else:
                        prob = random.random()
                        if prob < self.MINOR_EVENT["P"]:
                            start_night = self.night0 + TimeDelta(night, format="jd")
                            starts.append(start_night)
                            end_night = start_night + TimeDelta(self.MINOR_EVENT["length"], format="jd")
                            ends.append(end_night)
                            acts.append(self.MINOR_EVENT["level"])
                            night += self.MINOR_EVENT["length"] + 1
            night += 1
        self.downtime = np.array(
            list(zip(starts, ends, acts)),
            dtype=[("start", "O"), ("end", "O"), ("activity", "O")],
        )

    def total_downtime(self):
        """Return total downtime (in days).

        Returns
        -------
        total : `int`
            Total number of downtime days.
        """
        total = 0
        for td in self.downtime["end"] - self.downtime["start"]:
            total += td.jd
        return total


class UnscheduledDowntimeMoreY1Data(UnscheduledDowntimeData):
    def calc_sunrise_sets(self):
        """ """
        almanac = Almanac(mjd_start=self.start_time.mjd)
        year1_sunsets = np.where(
            (almanac.sunsets["night"] >= 0) & (almanac.sunsets["night"] < self.survey_length)
        )
        self.sunsets = almanac.sunsets[year1_sunsets]["sun_n12_setting"]
        self.sunrises = almanac.sunsets[year1_sunsets]["sun_n12_rising"]

    def make_data(self, end_of_start=380):
        """Configure the set of unscheduled downtimes.

        This function creates the unscheduled downtimes based on a set
        of probabilities of the downtime type occurance.

        The random downtime is calculated using the following
        probabilities:

        minor event : remainder of night and next day = 5/365 days
        e.g. power supply failure
        intermediate : 3 nights = 2/365 days e.g. repair band
        mechanism, rotator, hexapod, or shutter
        major event : 7 nights = 1/2*365 days
        catastrophic event : 14 nights = 1/3650 days e.g. replace a raft
        """

        self.calc_sunrise_sets()

        self.rng = np.random.default_rng(seed=self.seed)

        starts = []
        ends = []
        acts = []

        night_counted = np.zeros(len(self.sunsets))

        for night, (sunset, sunrise) in enumerate(zip(self.sunsets, self.sunrises)):
            prob = self.rng.random()
            hours_in_night = (sunrise - sunset) * 24.0
            if night_counted[night] == 1:
                continue

            if night < end_of_start:
                # Estimate a threshold probability of having some downtime
                # 50% at start, dropping until end_of_start, where it
                # should be .. 5%?
                nightly_threshold = 0.5 * (1 - night / (end_of_start + 45))
                if prob <= nightly_threshold:
                    # Generate an estimate of how long the downtime should be
                    prob_time = self.rng.gumbel(loc=1, scale=6, size=1)[0]
                    if prob_time >= hours_in_night:
                        prob_time = hours_in_night
                    if prob_time <= 1:
                        prob_time = 1.0
                    # And generate a starting time during the night for
                    # this event
                    tmax = hours_in_night - prob_time
                    if tmax <= 0:
                        starts.append(Time(sunset, format="mjd", scale="utc"))
                        ends.append(Time(sunrise, format="mjd", scale="utc"))
                        acts.append("Year1 Eng")
                    else:
                        offset = self.rng.uniform(low=sunset, high=sunset + tmax / 24.0)
                        starts.append(Time(offset, format="mjd", scale="utc"))
                        ends.append(Time(offset + prob_time / 24.0, format="mjd", scale="utc"))
                        acts.append("Year1 Eng")
                night_counted[night] = 1
                continue
        self.downtime = np.array(
            list(zip(starts, ends, acts)),
            dtype=[("start", "O"), ("end", "O"), ("activity", "O")],
        )

        regular_downtime = UnscheduledDowntimeData(
            start_time=self.start_time,
            seed=self.seed,
            start_of_night_offset=self.start_of_night_offset,
            survey_length=self.survey_length,
        )
        reg_dt = regular_downtime()
        self.downtime = np.concatenate([self.downtime, reg_dt])
        mjds = [row["start"].mjd for row in self.downtime]
        order = np.argsort(mjds)
        self.downtime = self.downtime[order]
