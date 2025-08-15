__all__ = ("BandSwapScheduler", "SimpleBandSched", "ComCamBandSched", "BandSchedUzy", "DateSwapBandScheduler")

import numpy as np
from astropy.time import Time

from rubin_scheduler.scheduler.utils import IntRounded


# Dictionary keys based on astropy times make me nervous.
# Let's use the dayobs's string instead.
def time_to_key(time):
    return time.isot.split("T")[0]


def key_to_time(key):
    return Time(f"{key}T12:00:00", scale="tai")


class BandSwapScheduler:
    """A simple way to schedule what band to load"""

    def __init__(self):
        pass

    def add_observation(self, observation):
        pass

    def __call__(self, conditions):
        """
        Returns
        -------
        list of strings for the bands that should be loaded
        """
        pass


class SimpleBandSched(BandSwapScheduler):
    """Swap the mounted bands depending on the lunar phase.

    This assumes we swap between just two sets of bandpasses, one
    for lunar phases where moon illumination at sunset < `illum_limit` and
    another for phases > `illum_limit`.

    Parameters
    ----------
    illum_limit
        The illumination limit to be compared to the `moon_illum`
        reported by the `Almanac`.
    """

    def __init__(self, illum_limit=10.0):
        self.illum_limit_ir = IntRounded(illum_limit)

    def __call__(self, conditions):
        if IntRounded(conditions.moon_phase_sunset) > self.illum_limit_ir:
            result = ["g", "r", "i", "z", "y"]
        else:
            result = ["u", "g", "r", "i", "z"]
        return result


class ComCamBandSched(BandSwapScheduler):
    """ComCam can only hold 3 bands at a time.

    Pretend we will cycle from ugr, gri, riz, izy
    depending on lunar phase.

    Parameters
    ----------
    loaded_band_groups : `tuple` (`tuple`)
        Groups of 3 bands, to be loaded at the same time.
        Multiple groups can be specified, to be swapped between at the
        boundaries of `illum_bins` in lunar phase.
    illum_bins : `np.ndarray`, (N,)
        Lunar illumination boundaries to define when to swap between
        different groups of bands within the `loaded_band_groups`.
        Lunar illumination ranges from 0 to 100.

    Notes
    -----
    If illum_bins = np.array([0, 50, 100]), then there should be
    two groups of bands to use -- one for use between 0 and 50 percent
    illumination, and another for use between 50 and 100 percent illumination.
    """

    def __init__(
        self,
        loaded_band_groups=(("u", "g", "r"), ("g", "r", "i"), ("r", "i", "z"), ("i", "z", "y")),
        illum_bins=np.arange(0, 100 + 1, 25),
    ):
        self.loaded_band_groups = loaded_band_groups
        self.illum_bins = illum_bins
        if isinstance(self.illum_bins, list):
            self.illum_bins = np.array(illum_bins)
        if len(illum_bins) - 1 > len(loaded_band_groups):
            raise ValueError("There are illumination bins with an " "undefined loaded_band_group")

    def __call__(self, conditions):
        moon_at_sunset = conditions.moon_phase_sunset
        try:
            if len(moon_at_sunset) > 0:
                moon_at_sunset = moon_at_sunset[0]
        except TypeError:
            pass
        indx = np.searchsorted(self.illum_bins, moon_at_sunset, side="left")
        indx = np.max([0, indx - 1])
        result = list(self.loaded_band_groups[indx])
        return result


class BandSchedUzy(BandSwapScheduler):
    """
    remove u in bright time. Alternate between removing z and y in
    dark time.

    Note, this might not work properly if we need to restart a bunch.
    So a more robust way of scheduling band loading might be in order.
    """

    def __init__(self, illum_limit=10.0):
        self.illum_limit_ir = IntRounded(illum_limit)
        self.last_swap = 0

        self.bright_time = ["g", "r", "i", "z", "y"]
        self.dark_times = [["u", "g", "r", "i", "y"], ["u", "g", "r", "i", "z"]]

    def __call__(self, conditions):
        if IntRounded(conditions.moon_phase_sunset) > self.illum_limit_ir:
            result = self.bright_time
        else:
            indx = self.last_swap % 2
            result = self.dark_times[indx]
            if result != conditions.mounted_bands:
                self.last_swap += 1
        return result


class DateSwapBandScheduler(BandSwapScheduler):
    """Swap specific bands on specific days, up until end_date, then
    fall back to the backup_filter_scheduler.

    This provides a way for simulations to use the specific bands that
    were in use on specific nights, as well as to incorporate knowledge
    of the upcoming scheduler, while still falling back to a reasonable
    filter swap schedule beyond the time the dates are precisely known.

    Parameters
    ----------
    swap_schedule
        Dictionary of times of the filter swaps, together with the filter
        complement loaded into the carousel.
        e.g. {"2025-08-20" : ['r', 'i', 'z', 'y']}
    end_date
        The Time after which the dictionary is no longer valid and the
        `backup_band_scheduler` should be used.
    backup_band_scheduler
        The more general band swap scheduler to use after the specific
        dates of the dictionary are exhausted. This should match the
        expected band scheduler for the Scheduler.
    """

    def __init__(
        self,
        swap_schedule: dict[str, list[str]] | None = None,
        end_date: Time | None = None,
        backup_band_scheduler: BandSwapScheduler = SimpleBandSched(illum_limit=40),
    ):
        previous_swap_schedule = {
            "2025-06-20": ["u", "g", "r", "i", "z"],
            "2025-07-01": ["g", "r", "i", "z"],
            "2025-07-04": ["g", "r", "i", "z", "y"],
            "2025-07-10": ["z"],
            "2025-07-11": ["g", "r", "i", "z", "y"],
            "2025-07-15": ["u", "g", "r", "i", "z"],
            "2025-07-28": [
                "u",
                "r",
                "i",
                "z",
            ],
            "2025-08-07": ["r", "i", "z", "y"],
            "2025-08-12": ["g", "r", "i", "z"],
        }
        previous_swap_times = np.sort(np.array([key_to_time(k) for k in previous_swap_schedule.keys()]))

        if swap_schedule is None:
            # Current estimate for the SV survey.
            # Subject to change, although past dates should match reality.
            swap_schedule = {
                "2025-08-12": ["g", "r", "i", "z"],
            }
        new_swap_times = np.sort(np.array([key_to_time(k) for k in swap_schedule.keys()]))

        # Join previous_swap_schedule and new schedule -
        # Use new swap_schedule if time overlaps with previous.
        keep_times = np.where(previous_swap_times < new_swap_times.min())[0]
        keep_keys = [time_to_key(t) for t in previous_swap_times[keep_times]]
        previous_swap_schedule = dict([(k, previous_swap_schedule[k]) for k in keep_keys])

        self.swap_schedule = previous_swap_schedule
        self.swap_schedule.update(swap_schedule)
        self.swap_schedule_times = np.sort(np.array([key_to_time(k) for k in self.swap_schedule.keys()]))

        if end_date is None:
            self.end_date = Time("2025-09-25T12:00:00")
        else:
            self.end_date = end_date

        if backup_band_scheduler is None:
            self.backup_band_scheduler = SimpleBandSched(illum_limit=40)
        else:
            self.backup_band_scheduler = backup_band_scheduler

    def __call__(self, conditions):
        current_time = Time(conditions.mjd, format="mjd", scale="tai")

        # Are we within the bounds of the scheduled swaps?
        if current_time < self.end_date:
            idx = np.where(current_time >= self.swap_schedule_times)[0][-1]
            return self.swap_schedule[time_to_key(self.swap_schedule_times[idx])]

        else:
            return self.backup_band_scheduler(conditions)
