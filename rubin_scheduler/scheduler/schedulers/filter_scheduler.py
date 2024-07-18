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
    """

    def __init__(self, illum_bins=np.arange(0, 100 + 1, 25),
                 filter_sets=(("u", "g", "r"),
                              ("g", "r", "i"),
                              ("r", "i", "z"),
                              ("i", "z", "y"))):
        self.illum_bins = illum_bins
        self.filter_sets = filter_sets

    def __call__(self, conditions):
        moon_at_sunset = conditions.moon_phase_sunset
        indx = np.searchsorted(self.illum_bins, moon_at_sunset, side="left")
        indx = np.max([0, indx - 1])
        result = list(self.filter_sets[indx])
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
