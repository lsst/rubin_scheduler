__all__ = ("ScriptedSurvey",)

import logging

import healpy as hp
import numpy as np

from rubin_scheduler.scheduler.surveys import BaseSurvey
from rubin_scheduler.scheduler.utils import ObservationArray, set_default_nside
from rubin_scheduler.utils import _angular_separation, _approx_ra_dec2_alt_az

log = logging.getLogger(__name__)


class ScriptedSurvey(BaseSurvey):
    """
    Take a set of scheduled observations and serve them up.

    Parameters
    ----------
    basis_functions : list of rubin_scheduler.scheduler.BasisFunction
        Basis functions to use. These are only used for computing
        survey feasibility. They do not contribute to the logic of
        how observations are selected. Basis functions that return
        HEALpix maps are ignored. Spatial masking is instead done
        with the `_check_alts_ha` method.
    id_start : `int` (1)
        The integer to start the "scripted id" field with. Bad things
        could happen if you have multiple scripted survey objects with
        the same scripted IDs.
    return_n_limit : `int` (10)
        The maximum number of observations to return. Set to high and
        your block of scheduled observations can run into twilight time.
    before_twi_check : `bool`
        Check if the returned observations have enough time to complete
        before twilight starts. (default True)
    filter_change_time : `float`
        The time needed to change filters. Default 120 seconds. Only
        used if before_twi_check is True.
    """

    def __init__(
        self,
        basis_functions,
        basis_weights=None,
        reward=1e6,
        ignore_obs=None,
        nside=None,
        detailers=None,
        id_start=1,
        return_n_limit=10,
        survey_name=None,
        before_twi_check=True,
        filter_change_time=120,
    ):
        """"""
        if nside is None:
            nside = set_default_nside()

        self.extra_features = {}
        self.nside = nside
        self.reward_val = reward
        self.reward = -np.inf
        self.id_start = id_start
        self.return_n_limit = return_n_limit
        self.filter_change_time = filter_change_time / 3600 / 24.0  # to days
        if basis_weights is None:
            self.basis_weights = np.zeros(len(basis_functions))
        else:
            if np.max(np.abs(basis_weights)) > 0:
                raise ValueError("Basis function weights should be zero for ScriptedSurvey objects.")
            if len(basis_weights) != len(basis_functions):
                raise ValueError("Length of Basis function weights should match length of basis_functions.")
            self.basis_weights = basis_weights
        self.before_twi_check = before_twi_check
        # Attribute to cache results into
        self.observations = None
        # Track when the last call to generate_observations_rough was.
        self.last_mjd = -1
        super(ScriptedSurvey, self).__init__(
            basis_functions=basis_functions,
            ignore_obs=ignore_obs,
            nside=nside,
            detailers=detailers,
            survey_name=survey_name,
        )
        self.clear_script()

        # Just to be clear that the script should
        # be setting scheduler_note, not the survey.
        self.scheduler_note = None

    def add_observations_array(self, observations_array_in, observations_hpid_in):
        if self.obs_wanted is not None:
            # toss out things that should be ignored
            to_ignore = np.in1d(observations_array_in["scheduler_note"], self.ignore_obs)
            observations_array = observations_array_in[~to_ignore]

            good = np.in1d(observations_hpid_in["ID"], observations_array["ID"])
            observations_hpid = observations_hpid_in[good]

            for feature in self.extra_features:
                self.extra_features[feature].add_observations_array(observations_array, observations_hpid)
            for bf in self.extra_basis_functions:
                self.extra_basis_functions[bf].add_observations_array(observations_array, observations_hpid)
            for bf in self.basis_functions:
                bf.add_observations_array(observations_array, observations_hpid)
            for detailer in self.detailers:
                detailer.add_observations_array(observations_array, observations_hpid)

            # If scripted_id, note, and filter match, then consider
            # the observation completed.
            completed = np.char.add(
                observations_array["scripted_id"].astype(str),
                observations_array["scheduler_note"],
            )
            completed = np.char.add(completed, observations_array["filter"])

            wanted = np.char.add(
                self.obs_wanted["scripted_id"].astype(str), self.obs_wanted["scheduler_note"]
            )
            wanted = np.char.add(wanted, self.obs_wanted["filter"])

            indx = np.in1d(wanted, completed)
            self.obs_wanted["observed"][indx] = True
            self.scheduled_obs = self.obs_wanted["mjd"][~self.obs_wanted["observed"]]

    def add_observation(self, observation, indx=None, **kwargs):
        """Check if observation matches a scripted observation"""
        if (self.obs_wanted is not None) & (np.size(self.obs_wanted) > 0):
            # From base class
            checks = [io not in str(observation["scheduler_note"]) for io in self.ignore_obs]
            if all(checks):
                for feature in self.extra_features:
                    self.extra_features[feature].add_observation(observation, **kwargs)
                for bf in self.basis_functions:
                    bf.add_observation(observation, **kwargs)
                for detailer in self.detailers:
                    detailer.add_observation(observation, **kwargs)
                self.reward_checked = False

                # Find the index
                indx = np.where(self.obs_wanted["scripted_id"] == observation["scripted_id"])[0]
                # If it matches scripted_id, note, and filter, mark it as
                # observed and update scheduled observation list.
                if indx.size > 0:
                    if (
                        (self.obs_wanted["scripted_id"][indx] == observation["scripted_id"])
                        & (self.obs_wanted["scheduler_note"][indx] == observation["scheduler_note"])
                        & (self.obs_wanted["filter"][indx] == observation["filter"])
                    ):
                        self.obs_wanted["observed"][indx] = True
                        self.scheduled_obs = self.obs_wanted["mjd"][~self.obs_wanted["observed"]]

    def calc_reward_function(self, conditions):
        """If there is an observation ready to go, execute it,
        otherwise, -inf"""
        observation = self.generate_observations_rough(conditions)
        if (observation is None) | (np.size(observation) == 0):
            self.reward = -np.inf
        else:
            self.reward = self.reward_val
        return self.reward

    def _slice2obs(self, obs_row):
        """take a slice and return a full observation object"""
        observation = ObservationArray()
        for key in [
            "RA",
            "dec",
            "filter",
            "exptime",
            "nexp",
            "scheduler_note",
            "target_name",
            "rotSkyPos",
            "rotTelPos",
            "flush_by_mjd",
            "scripted_id",
        ]:
            observation[key] = obs_row[key]
        return observation

    def _check_alts_ha(self, observation, conditions):
        """Given scheduled observations, check which ones can be
        done in current conditions.

        Parameters
        ----------
        observation : np.array
            An array of scheduled observations. Probably generated with
            rubin_scheduler.scheduler.utils.scheduled_observation
        """
        # distance to the moon
        d_to_moon = _angular_separation(
            observation["RA"], observation["dec"], conditions.moon_ra, conditions.moon_dec
        )

        # Just do a fast ra,dec to alt,az conversion.
        alt, az = _approx_ra_dec2_alt_az(
            observation["RA"],
            observation["dec"],
            conditions.site.latitude_rad,
            None,
            conditions.mjd,
            lmst=conditions.lmst,
        )
        HA = conditions.lmst - observation["RA"] * 12.0 / np.pi
        HA[np.where(HA > 24)] -= 24
        HA[np.where(HA < 0)] += 24
        in_range = np.where(
            (d_to_moon > observation["moon_min_distance"])
            & (alt < observation["alt_max"])
            & (alt > observation["alt_min"])
            & ((HA > observation["HA_max"]) | (HA < observation["HA_min"]))
            & (conditions.sun_alt < observation["sun_alt_max"])
        )[0]

        # Also check the alt,az limits given by the conditions object
        count = in_range * 0
        if conditions.tel_alt_limits is not None:
            for limits in conditions.tel_alt_limits:
                ir = np.where((alt[in_range] >= np.min(limits)) & (alt[in_range] <= np.max(limits)))[0]
                count[ir] += 1
            good = np.where(count > 0)[0]
            in_range = in_range[good]
        # Check against kinematic limits too
        ir = np.where(
            (alt[in_range] >= np.min(conditions.kinematic_alt_limits))
            & (alt[in_range] <= np.max(conditions.kinematic_alt_limits))
        )[0]
        in_range = in_range[ir]

        count = in_range * 0
        if conditions.tel_az_limits is not None:
            for limits in conditions.tel_az_limits:
                az_min = limits[0]
                az_max = limits[1]
                if np.abs(az_max - az_min) >= (2 * np.pi):
                    count += 1
                else:
                    az_range = (az_max - az_min) % (2 * np.pi)
                    ir = np.where((az[in_range] - az_min) % (2 * np.pi) <= az_range)[0]
                    count[ir] += 1
            good = np.where(count > 0)[0]
            in_range = in_range[good]
        if np.abs(conditions.kinematic_az_limits[1] - conditions.kinematic_az_limits[0]) < (2 * np.pi):
            az_range = (conditions.kinematic_az_limits[1] - conditions.kinematic_az_limits[0]) % (2 * np.pi)
            ir = np.where((az[in_range] - conditions.kinematic_az_limits[0]) % (2 * np.pi) <= az_range)[0]
            in_range = in_range[ir]

        # Check that filter needed is mounted
        good = np.isin(observation["filter"][in_range], conditions.mounted_filters)
        in_range = in_range[good]

        return in_range

    def _check_list(self, conditions):
        """Check to see if the current mjd is good"""
        observations = None
        if self.obs_wanted is not None:
            # Scheduled observations that are in the right time
            # window and have not been executed
            in_time_window = np.where(
                (self.mjd_start < conditions.mjd)
                & (self.obs_wanted["flush_by_mjd"] > conditions.mjd)
                & (~self.obs_wanted["observed"])
            )[0]

            if np.size(in_time_window) > 0:
                pass_checks = self._check_alts_ha(self.obs_wanted[in_time_window], conditions)
                matches = in_time_window[pass_checks]

                # Also check that the filters are mounted
                match2 = np.isin(self.obs_wanted["filter"][matches], conditions.mounted_filters)
                matches = matches[match2]

            else:
                matches = []

            if np.size(matches) > 0:
                # Do not return too many observations
                if np.size(matches) > self.return_n_limit:
                    matches = matches[0 : self.return_n_limit]
                observations = self.obs_wanted[matches]
                # Need to check that none of these are masked by basis
                # functions
                reward = 0
                for bf, weight in zip(self.basis_functions, self.basis_weights):
                    basis_value = bf(conditions)
                    reward += basis_value * weight
                # If reward is an array, then it's a HEALpy map and we
                # need to interpolate to the actual positions we want.
                # now to interpolate to the reward positions
                if np.size(reward) > 1:
                    reward_interp = hp.get_interp_val(
                        reward,
                        np.degrees(observations["RA"]),
                        np.degrees(observations["dec"]),
                        lonlat=True,
                    )
                    valid_reward = np.isfinite(reward_interp)
                    observations = observations[valid_reward]

        return observations

    def clear_script(self):
        """set an empty list to serve up"""
        self.obs_wanted = None
        self.mjd_start = None
        self.scheduled_obs = None

    def set_script(self, obs_wanted):
        """
        Parameters
        ----------
        obs_wanted : np.array
            The observations that should be executed. Needs to have
            columns with dtype names:
            Should be from lsst.sim.scheduler.utils.scheduled_observation
        mjds : np.array
            The MJDs for the observaitons, should be same length as
            obs_list
        mjd_tol : float (15.)
            The tolerance to consider an observation as still good to
            observe (min)
        """

        self.obs_wanted = obs_wanted

        self.obs_wanted.sort(order=["mjd", "filter"])
        # Give each desired observation a unique "scripted ID". To be used for
        # matching and logging later.
        self.obs_wanted["scripted_id"] = np.arange(self.id_start, self.id_start + np.size(self.obs_wanted))
        # Update so if we set the script again the IDs will not be reused.
        self.id_start = np.max(self.obs_wanted["scripted_id"]) + 1

        self.mjd_start = self.obs_wanted["mjd"] - self.obs_wanted["mjd_tol"]
        # Here is the attribute that core scheduler checks to
        # broadcast scheduled observations in the conditions object.
        self.scheduled_obs = self.obs_wanted["mjd"]

    def generate_observations_rough(self, conditions):
        # if we have already called for this mjd, no need to repeat.
        if self.last_mjd == conditions.mjd:
            return self.observations

        observations = self._check_list(conditions)
        if observations is None:
            self.observations = []
            self.last_mjd = conditions.mjd
            return self.observations

        n_filter_changes = np.sum(observations[1:]["filter"] == observations[:-1]["filter"])

        # If we want to ensure the observations can be completed
        # before twilight starts
        if self.before_twi_check:
            # Note that if detailers are adding lots of exposures, this
            # calculation has the potential to not be right at all.
            # Also assumes slew time is negligible.
            exptime_needed = np.sum(observations["exptime"]) / 3600.0 / 24.0  # to days
            filter_change_needed = n_filter_changes * self.filter_change_time
            tot_time_needed = exptime_needed + filter_change_needed
            time_before_twi = conditions.sun_n18_rising - conditions.mjd
            # Not enough time, wipe out the observations
            if tot_time_needed > time_before_twi:
                self.observations = []
                self.last_mjd = conditions.mjd
                return self.observations

        self.observations = [self._slice2obs(obs) for obs in observations]
        self.last_mjd = conditions.mjd

        return self.observations
