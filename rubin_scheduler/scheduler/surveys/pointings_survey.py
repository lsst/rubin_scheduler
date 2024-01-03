__all__ = ["PointingsSurvey"]

import healpy as hp
import numpy as np

from rubin_scheduler.scheduler.detailers import ParallacticRotationDetailer
from rubin_scheduler.scheduler.utils import set_default_nside
from rubin_scheduler.skybrightness_pre import dark_m5

from .base_survey import BaseSurvey


class PointingsSurvey(BaseSurvey):
    """Select a potential observation from a pre defined list.

    Note the nside resolution must be set higher than the smallest
    spacing between potential observations.
    """

    def __init__(
        self,
        basis_functions,
        basis_weights=None,
        ignore_obs="dummy",
        nside=None,
        detailers=None,
        id_start=1,
        return_n_limit=10,
        survey_name=None,
        method_weights=None,
        fiducial_FWHMEff=0.7,
        gap_min=25.0,
        track_notes_ngoal=None,
    ):
        """"""
        if nside is None:
            nside = set_default_nside()

        self.extra_features = {}
        self.nside = nside
        self.reward = -np.inf
        self.id_start = id_start
        self.return_n_limit = return_n_limit
        if basis_weights is None:
            self.basis_weights = np.zeros(len(basis_functions))
        else:
            if np.max(np.abs(basis_weights)) > 0:
                raise ValueError("Basis function weights should be zero for PointingsSurvey objects.")
            if len(basis_weights) != len(basis_functions):
                raise ValueError("Length of Basis function weights should match length of basis_functions.")
            self.basis_weights = basis_weights
        super(PointingsSurvey, self).__init__(
            basis_functions=basis_functions,
            ignore_obs=ignore_obs,
            nside=nside,
            detailers=detailers,
            survey_name=survey_name,
        )

        self.last_computed_reward = -1.0

        if method_weights is None:
            self.method_weights = {
                "visit_gap": 1.0,
                "balance_revisit": 1.0,
                "m5diff": 1.0,
                "slew_time": -1.0,
            }
        else:
            self.method_weights = method_weights

        self.fiducial_FWHMEff = fiducial_FWHMEff
        self.gap_min = gap_min / 60.0 / 24.0  # to days
        self.dark_map = None
        self.tracking_notes = {}
        self.track_notes_ngoal = track_notes_ngoal
        if track_notes_ngoal is not None:
            for key in track_notes_ngoal:
                self.tracking_notes[key] = 0

        # If there's no detailers, add one to set rotation to
        # Parallactic angle
        if detailers is None:
            self.detailers = [ParallacticRotationDetailer()]
        else:
            self.detailers = detailers

    def set_observations(self, observations):
        """ """
        if len(np.unique(observations["note"])) < len(observations):
            raise ValueError("observations['note'] values are not unique")

        self.observations = observations
        self.zeros = np.zeros(self.observations.size, dtype=float)

        # Arrays to track progress
        self.n_obs = np.zeros(self.zeros.size, dtype=int)
        self.last_observed = np.zeros(self.zeros.size, dtype=float)

        # Make mapping so it's fast to compute reward later
        # Note that user must be careful to keep sequence names unique
        self.sequence_mapping = {}
        for key in self.tracking_notes:
            self.sequence_mapping[key] = []
        for i, obs in enumerate(observations):
            for key in self.sequence_mapping:
                if key in obs["note"]:
                    self.sequence_mapping[key].append(i)

    def add_observation(self, observation, indx=None):
        # Check for a nore match
        indx = np.where(observation["note"] == self.observations["note"])[0]
        # Tracking arrays
        self.n_obs[indx] += 1
        self.last_observed[indx] = observation["mjd"]

        # If we are tracking n observations of some type:
        for key in self.tracking_notes:
            if key in observation["note"]:
                self.tracking_notes[key] += 1

    def add_observations_array(self, observations_array_in, observations_hpid_in):
        for unote in np.unique(observations_array_in["note"]):
            matching = np.where(observations_array_in["note"] == unote)[0]
            indx = np.where(self.observations["note"] == unote)[0]
            self.n_obs[indx] += np.size(matching)
            self.last_observed[indx] = observations_array_in["mjd"][matching].max()

            for key in self.tracking_notes:
                if key in unote:
                    self.tracking_notes[key] += np.size(matching)

    def calc_reward_function(self, conditions):
        """
        Parameters
        ----------
        conditions : rubin_scheduler.scheduler.features.Conditions object

        Returns
        -------
        reward : float (or array)

        """
        if self._check_feasibility(conditions):
            self.reward = 0
            indx = np.arange(hp.nside2npix(self.nside))
            for bf, weight in zip(self.basis_functions, self.basis_weights):
                basis_value = bf(conditions, indx=indx)
                self.reward += basis_value * weight

            if np.size(self.reward) > 1:
                # Interpolate the reward function to the positions in self.observations
                self.reward = hp.get_interp_val(
                    self.reward,
                    np.degrees(self.observations["RA"]),
                    np.degrees(self.observations["dec"]),
                    lonlat=True,
                )

            # Things are masked that should be, now to use methods to
            # compute desireability
            # In theory, could track where the reward is already
            # NaN and not bother doing extra computations.
            for key in self.method_weights:
                self.reward += self.method_weights[key] * getattr(self, key)(conditions)

        else:
            # If we don't pass feasability
            self.reward = -np.inf

        self.reward_checked = True
        return np.nanmax(self.reward)

    def generate_observations_rough(self, conditions):
        """ """
        # If the reward function hasn't been updated with the
        # latest info, calculate it
        if not self.reward_checked:
            self.reward = self.calc_reward_function(conditions)
        max_indx = np.where(self.reward == np.nanmax(self.reward))[0].min()
        obs = self.observations[max_indx].copy().reshape(1)

        return [obs]

    def visit_gap(self, conditions):
        """Enforce a minimum visit gap."""
        diff = conditions.mjd - self.last_observed
        too_soon = np.where(diff < self.gap_min)[0]
        result = self.zeros.copy()
        # Using NaN makes it a hard limit
        # could have a weight and just subtract from the reward
        result[too_soon] = np.nan
        return result

    def balance_revisit(self, conditions):
        """Code to balance revisiting different targets."""
        sum_obs = np.sum(self.n_obs)
        result = np.floor(1.0 + self.n_obs / sum_obs)
        result[np.where(self.n_obs == 0)] = 1
        return result

    def slew_time(self, conditions):
        # Interpolate the conditions slewtime map to our positions
        # Could do something more sophisticated, but this is probably fine.
        result = hp.get_interp_val(
            conditions.slewtime,
            np.degrees(self.observations["RA"]),
            np.degrees(self.observations["dec"]),
            lonlat=True,
        )
        # Could do a filter check here and add a penalty for changing filters

        return result

    def sequence_boost(self, conditions):
        """Boost the reward if a sequence is incomplete."""
        result = self.zeros.copy()
        for key in self.sequence_mapping:
            result[self.sequence_mapping[key]] = self.tracking_notes[key] % self.track_notes_ngoal[key]

        return result

    def _dark_map(self, conditions):
        """Generate the dark map if needed

        Constructs self.dark_map which is a dict with
        keys of filtername and values of HEALpix arrays
        that are the darkest expected 5-sigma limiting depth
        expected for that region of sky
        """
        self.dark_map = {}
        for filtername in np.unique(self.observations["filter"]):
            self.dark_map[filtername] = dark_m5(
                conditions.dec, filtername, conditions.site.latitude_rad, self.fiducial_FWHMEff
            )

    def m5diff(self, conditions):
        """Compute difference between current 5-sigma limiting depth and
        the depth at the same coordinates in "ideal" conditions.
        """
        if self.dark_map is None:
            self._dark_map(conditions)
        result = np.zeros(self.observations.size)
        for filtername in np.unique(self.observations["filter"]):
            indx = np.where(self.observations["filter"] == filtername)[0]
            diff_map = conditions.m5_depth[filtername] - self.dark_map[filtername]
            result[indx] = hp.get_interp_val(
                diff_map,
                np.degrees(self.observations["RA"][indx]),
                np.degrees(self.observations["dec"][indx]),
                lonlat=True,
            )

        return result
