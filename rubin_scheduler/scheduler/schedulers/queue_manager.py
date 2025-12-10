__all__ = ("BaseQueueManager",)

import healpy as hp
import numpy as np

from rubin_scheduler.scheduler.utils import IntRounded


class BaseQueueManager:
    """Class for managing a queue of desired observations.

    NOTE: If using masking basis functions, one should
    ensure Survey objects are using the same masks, else
    one will end up in a loop where surveys propose visits
    that the QueueManager always rejects.

    Parameters
    ----------
    detailers : `list`
        List of detailers to apply to observations as they are
        selected from the queue. Mostly for setting camera
        rotation angles dynamically, or skipping observations
        that have become clouded out or gotten close to zenith.
    basis_functions : `list`
        List of basis functions. Assumed to be only used for
        masking, thus no weights for the basis functions.

    """

    def __init__(self, detailers=None, basis_functions=None):
        if detailers is None:
            self.detailers = []
        else:
            self.detailers = detailers
        if basis_functions is None:
            self.basis_functions = []
        else:
            self.basis_functions = basis_functions
        self.desired_observations_array = None
        # Array to track which desired_observations_array
        # have been observed
        self.need_observing = False

    def flush_queue(self):
        self.desired_observations_array = None
        self.need_observing = False

    def set_queue(self, observation_array):
        self.desired_observations_array = observation_array
        self.need_observing = np.ones(observation_array.size, dtype=bool)

    def add_observation(self, observation):
        if self.desired_observations_array is not None:
            match_indx = np.where(self.desired_observations_array["target_id"] == observation["target_id"])
            self.need_observing[match_indx] = False

    def add_observations_array(self, observation_array):
        if self.desired_observations_array is not None:
            indx = np.isin(self.desired_observations_array["target_id"], observation_array["target_id"])
            self.need_observing[indx] = False

    def compute_reward(self, conditions):
        reward = 0
        for bf in self.basis_functions:
            reward += bf(conditions)
        return reward

    def find_valid_indx(self, conditions):
        """Return the indices of desired_observations_array
        that can be observed
        """
        valid = np.where(self.need_observing)[0]

        # Compute reward from basis functions
        reward = self.compute_reward(conditions)
        # If reward is an array, then it's a HEALpy map and we
        # need to interpolate to the actual positions we want.
        # now to interpolate to the reward positions
        if np.size(reward) > 1:
            reward_interp = hp.get_interp_val(
                reward,
                np.degrees(self.desired_observations_array["RA"][valid]),
                np.degrees(self.desired_observations_array["dec"][valid]),
                lonlat=True,
            )
            sub_valid_reward = np.isfinite(reward_interp)
            valid = np.ravel(valid[sub_valid_reward])

        return valid

    def _check_queue_mjd_only(self, mjd):
        result = False
        if np.sum(self.need_observing) > 0:
            if np.any(
                IntRounded(mjd)
                < IntRounded(self.desired_observations_array[self.need_observing]["flush_by_mjd"])
            ) | (np.any(self.desired_observations_array[self.need_observing]["flush_by_mjd"] == 0)):
                result = True
        return result

    def request_observation(self, conditions, whole_queue=False, n_return=1):
        """
        Parameters
        ----------
        conditions : ``
            Conditions object
        whole_queue : `bool`
            If True, return all desired observations with no check if
            they are accessible.
        n_return : `int`
            Number of observations to return.
        """

        # Everything that needs observing
        if whole_queue:
            return self.desired_observations_array[self.need_observing]

        indx = self.find_valid_indx(conditions)

        if indx.size > n_return:
            indx = indx[0:n_return]

        result = self.desired_observations_array[indx].copy()
        if np.size(result) > 0:
            for det in self.detailers:
                result = det(result, conditions)

        return result

    def return_active_queue(self):
        """Return array of observations that are waiting to be executed"""
        return self.desired_observations_array[self.need_observing]
