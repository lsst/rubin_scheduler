__all__ = ("SummitWrapper",)

import copy

import numpy as np

from rubin_scheduler.utils import _ra_dec2_hpid


class SummitWrapper:
    """Wrap a core scheduler so observations can be requested
    and are assumed to be completed.
    """

    def __init__(self, core_scheduler):
        self.core_scheduler = core_scheduler
        self.conditions = None
        self.clear_ahead()

    def clear_ahead(self):
        """Reset the ahead scheduler and pending observations"""
        self.ahead_scheduler = copy.deepcopy(self.core_scheduler)
        self.requested_but_unadded_ids = []
        self.requested_but_unadded_obs = []
        # Have we added observations so
        # self.core_scheduler and self.ahead_scheduler
        # are out of sync?
        self.need_reset = False

    def add_observation(self, observation):

        self.core_scheduler.add_observation(observation)

        # Assume everything up to the ID has been observed.
        # Should be ok to add out of order, as long as everything up
        # to the largest ID is added before request_observation is called.
        indx = np.searchsorted(self.requested_but_unadded_ids, observation["ID"], side="right")
        # Should this delete everything up to the indx?
        # I think that's ok if we assume things will get added in order
        if indx.size > 0:
            self.requested_but_unadded_ids = self.requested_but_unadded_ids[indx:]
            self.requested_but_unadded_obs = self.requested_but_unadded_obs[indx:]

        self.need_reset = True

    def update_conditions(self, conditions):

        self.conditions = conditions
        self.ahead_scheduler.update_conditions(conditions)

    def _fill_obs_values(self, observation):
        """Fill in values of an observation assuming it will be
        observed now or very soon
        """

        # Nearest neighbor from conditions maps
        hpid = _ra_dec2_hpid(self.conditions.nside, observation["RA"], observation["dec"])

        observation["mjd"] = self.conditions.mjd
        observation["FWHMeff"] = self.conditions.fwhm_eff[observation["band"]][hpid]
        observation["airmass"] = self.conditions.airmass[hpid]
        observation["fivesigmadepth"] = self.conditions.m5_depth[observation["band"]][hpid]
        observation["night"] = self.conditions.night

        return observation

    def request_observation(self):
        """Request an observation, assuming previously requested
        observations were successfully observed.
        """

        if self.need_reset:
            self.ahead_scheduler = copy.deepcopy(self.core_scheduler)
            for obs in self.requested_but_unadded_obs:
                self.ahead_scheduler.add_observation(obs)
            self.ahead_scheduler.update_conditions(self.conditions)
            self.need_reset = False

        result_plain = self.ahead_scheduler.request_observation()

        obs_filled = self._fill_obs_values(result_plain.copy())
        self.requested_but_unadded_ids.append(obs_filled["ID"])
        self.requested_but_unadded_obs.append(obs_filled)

        # Add requested observation to the ahead
        # scheduler, so if we call request_observation again,
        # it will assume this has been done.
        self.ahead_scheduler.add_observation(obs_filled)

        return result_plain
