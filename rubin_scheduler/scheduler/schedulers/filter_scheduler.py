__all__ = ("FilterSwapScheduler", "SimpleFilterSched", "ComCamFilterSched", "FilterSchedUzy")

import numpy as np

from rubin_scheduler.scheduler.utils import IntRounded


class FilterSwapScheduler:
    """A simple way to schedule what filter to load"""

    def __init__(self):
        pass

    def add_observation(self, observation):
        pass

    def __call__(self, conditions):
        """
        Returns
        -------
        list of strings for the filters that should be loaded
        """
        pass


class SimpleFilterSched(FilterSwapScheduler):
    def __init__(self, illum_limit=10.0):
        self.illum_limit_ir = IntRounded(illum_limit)

    def __call__(self, conditions):
        if IntRounded(conditions.moon_phase_sunset) > self.illum_limit_ir:
            result = ["g", "r", "i", "z", "y"]
        else:
            result = ["u", "g", "r", "i", "z"]
        return result


class ComCamFilterSched(FilterSwapScheduler):
    """ComCam can only hold 3 filters at a time.

    Pretend we will cycle from ugr, gri, riz, izy
    depending on lunar phase.

    Parameters
    ----------
    loaded_filter_groups : `tuple` (`tuple`)
        Groups of 3 filters, to be loaded at the same time.
        Multiple groups can be specified, to be swapped between at the
        boundaries of `illum_bins` in lunar phase.
    illum_bins : `np.ndarray`, (N,)
        Lunar illumination boundaries to define when to swap between
        different groups of filters within the `loaded_filter_groups`.
        Lunar illumination ranges from 0 to 100.

    Notes
    -----
    If illum_bins = np.array([0, 50, 100]), then there should be
    two groups of filters to use -- one for use between 0 and 50 percent
    illumination, and another for use between 50 and 100 percent illumination.
    """

    def __init__(
        self,
        loaded_filter_groups=(("u", "g", "r"), ("g", "r", "i"), ("r", "i", "z"), ("i", "z", "y")),
        illum_bins=np.arange(0, 100 + 1, 25),
    ):
        self.loaded_filter_groups = loaded_filter_groups
        self.illum_bins = illum_bins
        if isinstance(self.illum_bins, list):
            self.illum_bins = np.array(illum_bins)
        if len(illum_bins) - 1 > len(loaded_filter_groups):
            raise ValueError("There are illumination bins with an " "undefined loaded_filter_group")

    def __call__(self, conditions):
        moon_at_sunset = conditions.moon_phase_sunset
        try:
            if len(moon_at_sunset) > 0:
                moon_at_sunset = moon_at_sunset[0]
        except TypeError:
            pass
        indx = np.searchsorted(self.illum_bins, moon_at_sunset, side="left")
        indx = np.max([0, indx - 1])
        result = list(self.loaded_filter_groups[indx])
        return result


class FilterSchedUzy(FilterSwapScheduler):
    """
    remove u in bright time. Alternate between removing z and y in
    dark time.

    Note, this might not work properly if we need to restart a bunch.
    So a more robust way of scheduling filter loading might be in order.
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
            if result != conditions.mounted_filters:
                self.last_swap += 1
        return result
