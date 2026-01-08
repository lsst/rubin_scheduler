__all__ = ("SummitWrapper",)

import copy

import numpy as np

from rubin_scheduler.scheduler.utils import IntRounded
from rubin_scheduler.utils import _ra_dec2_hpid


class SummitWrapper:
    """Wrap a core scheduler so observations can be requested
    and are assumed to be completed.
    """

    def __init__(self, core_scheduler):
        self.core_scheduler = core_scheduler
        self.conditions = None
        self.clear_ahead()

        # For compatibility with sim_runner.
        # Not tracking flushed observations
        self.flushed = 0

    def clear_ahead(self):
        """Reset the ahead scheduler and pending observations"""
        self.ahead_scheduler = copy.deepcopy(self.core_scheduler)
        self.requested_but_unadded_ids = []
        self.requested_but_unadded_obs = []
        # Have we added observations so
        # self.core_scheduler and self.ahead_scheduler
        # are out of sync?
        self.need_reset = False

    def flush_queue(self):
        self.core_scheduler.flush_queue()
        self.clear_ahead()

    def add_observation(self, observation):

        self.core_scheduler.add_observation(observation)

        # If this observation is in the core.queue, need to
        # remove it from there.
        if len(self.core_scheduler.queue) > 0:
            # Ugh, kind of a pain that observations have
            # been converted to a list.
            target_ids = [obs["target_id"] for obs in self.core_scheduler.queue]
            match = np.where(target_ids == observation["target_id"])[0].max()
            if np.size(match) > 0:
                del self.core_scheduler.queue[match.astype(int)]

        # Assume everything up to the ID has been observed.
        # Should be ok to add out of order, as long as everything up
        # to the largest ID is added before request_observation is called.
        indx = np.searchsorted(
            self.requested_but_unadded_ids, observation["target_id"].view(np.ndarray).max(), side="right"
        )

        # Should this delete everything up to the indx?
        # I think that's ok if we assume things will get added in order
        if indx.size > 0:
            self.requested_but_unadded_ids = self.requested_but_unadded_ids[indx:]
            self.requested_but_unadded_obs = self.requested_but_unadded_obs[indx:]

        self.need_reset = True

    def _check_queue_mjd_only(self, mjd):
        """
        Check if there are things in the queue that can be executed
        using only MJD and not full conditions.
        This is primarily used by sim_runner to reduce calls calculating
        updated conditions when they are not needed.
        """
        result = False
        if len(self.ahead_scheduler.queue) > 0:
            if (IntRounded(mjd) < IntRounded(self.ahead_scheduler.queue[0]["flush_by_mjd"])) | (
                self.ahead_scheduler.queue[0]["flush_by_mjd"] == 0
            ):
                result = True
        return result

    def update_conditions(self, conditions):

        self.conditions = conditions
        self.ahead_scheduler.update_conditions(conditions)
        self.core_scheduler.update_conditions(conditions)

    def _fill_obs_values(self, observation):
        """Fill in values of an observation assuming it will be
        observed now or very soon
        """

        # Nearest neighbor from conditions maps
        hpid = _ra_dec2_hpid(self.conditions.nside, observation["RA"], observation["dec"])

        # XXX--Need to go through and formally list the columns
        # that are minimally required for adding observations to Features.
        # This seems to work, but nothing stopping someone from asking for
        # something like moon phase and then it will fail. Maybe
        # set all un-defined columns to np.nan so it's clear they
        # can't be used unless this method is updated.
        observation["mjd"] = self.conditions.mjd
        observation["FWHMeff"] = self.conditions.fwhm_eff[observation["band"][0]][hpid]
        observation["airmass"] = self.conditions.airmass[hpid]
        observation["fivesigmadepth"] = self.conditions.m5_depth[observation["band"][0]][hpid]
        observation["night"] = self.conditions.night

        return observation

    def request_observation(self, mjd=None):
        """Request an observation, assuming previously requested
        observations were successfully observed.
        """

        if mjd is None:
            mjd = self.conditions.mjd

        if self.need_reset:
            self.ahead_scheduler = copy.deepcopy(self.core_scheduler)
            for obs in self.requested_but_unadded_obs:
                self.ahead_scheduler.add_observation(obs)
            self.need_reset = False
            self.ahead_scheduler.update_conditions(self.conditions)

        # If we fill the queue, need to add that to the core scheduler
        # so it knows about it

        # If ahead has things in the queue, it'll just pop it off
        if len(self.ahead_scheduler.queue) > 0:
            result_plain = self.ahead_scheduler.request_observation(mjd=mjd)
        else:
            # Now, we have either refilled the queue and popped one, or
            # generated a one-off and have an empty queue.
            result_plain = self.ahead_scheduler.request_observation(mjd=mjd)
            self.core_scheduler.queue.append(result_plain.copy())
            if len(self.ahead_scheduler.queue) > 0:
                self.core_scheduler.queue += self.ahead_scheduler.queue

        obs_filled = self._fill_obs_values(result_plain.copy())
        self.requested_but_unadded_ids.append(obs_filled["target_id"].view(np.ndarray).max())
        self.requested_but_unadded_obs.append(obs_filled)

        # Add requested observation to the ahead
        # scheduler, so if we call request_observation again,
        # it will know this has been done.
        self.ahead_scheduler.add_observation(obs_filled)

        return result_plain
