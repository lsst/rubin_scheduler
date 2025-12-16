__all__ = ("BaseQueueManager",)

import healpy as hp
import numpy as np

from rubin_scheduler.scheduler.utils import IntRounded, ObservationArray


class BaseQueueManager:
    """Class for managing a queue of desired observations.

    NOTE: If using masking basis functions, one should
    ensure Survey objects are using the same masks as the
    QueueManager, else one will end up in a loop where
    surveys propose visits that the QueueManager always rejects.
    E.g., if the surveys don't mask clouds, all their observations
    can be rejected by the QueueManager.

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
        self.desired_observations_array = ObservationArray(n=0)
        # Array to track which desired_observations_array
        # have been observed
        self.need_observing = []

    def flush_queue(self):
        self.desired_observations_array = ObservationArray(n=0)
        self.need_observing = []

    def set_queue(self, observation_array):
        self.desired_observations_array = observation_array
        self.need_observing = np.ones(observation_array.size, dtype=bool)

    def add_observation(self, observation, **kwargs):
        if self.desired_observations_array is not None:
            match_indx = np.where(self.desired_observations_array["target_id"] == observation["target_id"])[0]
            if np.size(match_indx) > 0:
                self.need_observing[match_indx] = False
        for bf in self.basis_functions:
            bf.add_observation(observation, **kwargs)
        for detailer in self.detailers:
            detailer.add_observation(observation, **kwargs)

    def add_observations_array(self, observation_array, observations_hpid_in):
        if self.desired_observations_array is not None:
            indx = np.isin(self.desired_observations_array["target_id"], observation_array["target_id"])
            if np.size(indx) > 0:
                self.need_observing[indx] = False

            good = np.isin(observations_hpid_in["ID"], observation_array["ID"])
            observations_hpid = observations_hpid_in[good]
            for bf in self.basis_functions:
                bf.add_observations_array(observation_array, observations_hpid)
            for detailer in self.detailers:
                detailer.add_observations_array(observation_array, observations_hpid)

    def compute_reward(self, conditions):
        reward = 0
        for bf in self.basis_functions:
            reward += bf(conditions)
        return reward

    def _check_queue_mjd_only(self, mjd, conditions):
        """
        Check if there are things in the queue that can be executed
        using only MJD and not full conditions.
        This is primarily used by sim_runner to reduce calls calculating
        updated conditions when they are not needed.
        """

        # With queue manager doing cloud dodging, need full conditions
        # now. Go ahead and use stale conditions, but update the
        # mjd so things get flushed if needed.
        valid_obs = self.find_valid_observations_indx(conditions, mjd=mjd)

        return np.size(valid_obs) > 0

    def find_valid_observations_indx(self, conditions, mjd=None):
        """Return the indices of desired_observations_array
        that can be observed
        """

        if mjd is None:
            mjd = conditions.mjd

        # Still valid date
        mjd_ok = np.where(
            (IntRounded(mjd) < IntRounded(self.desired_observations_array["flush_by_mjd"]))
            | (self.desired_observations_array["flush_by_mjd"] == 0)
        )[0]

        valid = np.where(self.need_observing)[0]

        valid = np.intersect1d(mjd_ok, valid)

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
            indx = np.where(self.need_observing)[0]
        else:
            indx = self.find_valid_observations_indx(conditions)
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
