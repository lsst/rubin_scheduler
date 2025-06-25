__all__ = ["PointingsSurvey"]

import healpy as hp
import numpy as np
import pandas as pd

from rubin_scheduler.scheduler.detailers import ParallacticRotationDetailer
from rubin_scheduler.scheduler.utils import IntRounded, ObservationArray
from rubin_scheduler.skybrightness_pre import dark_m5
from rubin_scheduler.utils import _angular_separation, _approx_ra_dec2_alt_az

from .base_survey import BaseSurvey


class PointingsSurvey(BaseSurvey):
    """Survey object for managing a set list of potential pointings
    without specified observing times.

    Does not follow the usual Survey class API by not using
    BasisFunction objects -- this makes it unsuitable for use for Schedulers
    which use generic masks to avoid observing in out of bounds areas.


    Parameters
    ----------
    observations : `np.array`
        An array of observations, from e.g.,
        rubin_scheduler.scheduler.utils.ObservationArray
        expect "RA", "dec", and "note" to be filled, other columns ignored.
    gap_min : `float`
        The minimum gap to force between observations of the same
        spot (minutes)
    alt_min : `float`
        Altitude limit of the telescope (degrees). Default 20.
    alt_max : `float`
        Altitude limit of the telescope (degrees). Default 85.
    ha_max, ha_min : `float` (4,-4)
        hour angle limits (hours). Applied to all observations.
        Default 4,-4.
    weights : `dict`
        Dictionary with keys of method names and values of floats.
        Default of None uses {"visit_gap": 1.0, "balance_revisit": 1.0,
                              "m5diff": 1.0,
                              "wind_limit": 1.0, "slew_time": -1.0,
                              "ha_limit": 0, "alt_limit": 0,
                              "moon_limit": 0}
    wind_speed_maximum : `float`
        The maximum wind (m/s), mask any targets that would take more
        wind than that. Default 100 m/s.
    fiducial_FWHMEff : `float`
        A fiducial seeing value to use when computing dark sky depth.
        Default 0.7 arcsec.
    sun_alt_limit : `float`
        Have survey as infeasible when sun altitude is above the limit.
        Default -12 (degrees).
    track_notes_ngoal : `dict`
        If there are observations that should be tracked together
        (e.g., a sequence that should be observed together). Dict
        with keys of str and values of int.
    """

    def __init__(
        self,
        observations,
        gap_min=25.0,
        moon_dist_limit=30.0,
        weights=None,
        alt_max=85.0,
        alt_min=20.0,
        detailers=None,
        ha_max=4,
        ha_min=-4,
        wind_speed_maximum=100,
        fiducial_FWHMEff=0.7,
        sun_alt_limit=-12.0,
        track_notes_ngoal=None,
    ):
        # Not doing a super here, don't want to even have an nside defined.

        # Check that observations["scheduler_note"] are unique, otherwise
        # incoming observations will get double-counted
        if np.size(np.unique(observations["scheduler_note"])) != np.size(observations):
            raise ValueError("observations['scheduler_note'] values are not unique")

        self.observations = observations
        self.gap_min = gap_min / 60.0 / 24.0  # to days
        self.moon_dist_limit = np.radians(moon_dist_limit)
        self.alt_max = np.radians(alt_max)
        self.alt_min = np.radians(alt_min)
        self.zeros = self.observations["RA"] * 0.0
        self.wind_speed_maximum = wind_speed_maximum
        self.dark_map = None
        self.fiducial_FWHMEff = fiducial_FWHMEff
        self.sun_alt_limit = np.radians(sun_alt_limit)
        self.tracking_notes = {}
        self.track_notes_ngoal = track_notes_ngoal
        if track_notes_ngoal is not None:
            for key in track_notes_ngoal:
                self.tracking_notes[key] = 0
        # Make mapping so it's fast to compute reward later
        # Note that user must be careful to keep sequence names unique
        self.sequence_mapping = {}
        for key in self.tracking_notes:
            self.sequence_mapping[key] = []
        for i, obs in enumerate(observations):
            for key in self.sequence_mapping:
                if key in obs["scheduler_note"]:
                    self.sequence_mapping[key].append(i)

        # convert hour angle limits to radians and 0-2pi
        self.ha_max = (np.radians(ha_max * 360 / 24) + 2 * np.pi) % (2 * np.pi)
        self.ha_min = (np.radians(ha_min * 360 / 24) + 2 * np.pi) % (2 * np.pi)

        # Arrays to track progress
        self.n_obs = np.zeros(self.zeros.size, dtype=int)
        self.last_observed = np.zeros(self.zeros.size, dtype=float)

        self.last_computed_reward = -1.0

        if weights is None:
            self.weights = {
                "visit_gap": 1.0,
                "balance_revisit": 1.0,
                "m5diff": 1.0,
                "slew_time": -1.0,
                "wind_limit": 0.0,
                "ha_limit": 0,
                "alt_limit": 0,
                "moon_limit": 0,
            }
        else:
            self.weights = weights

        self.scheduled_obs = None
        # If there's no detailers, add one to set rotation to
        # Parallactic angle
        if detailers is None:
            self.detailers = [ParallacticRotationDetailer()]
        else:
            self.detailers = detailers

    def _check_feasibility(self, conditions):
        """Check if the survey is feasable in the current conditions"""
        # Simple feasability check.
        result = True

        # if the sun is too high
        if IntRounded(conditions.sun_alt) > IntRounded(self.sun_alt_limit):
            return False

        reward = self.calc_reward_function(conditions)
        if not np.isfinite(reward):
            result = False
        return result

    def calc_reward_function(self, conditions):
        """Compute reward function using methods set by `weights`
        dict on init."""
        if self.last_computed_reward != conditions.mjd:
            self.alt, self.az = _approx_ra_dec2_alt_az(
                self.observations["RA"],
                self.observations["dec"],
                conditions.site.latitude_rad,
                conditions.site.longitude_rad,
                conditions.mjd,
                lmst=conditions.lmst,
            )

            self.ha = np.radians(conditions.lmst * 360.0 / 24.0) - self.observations["RA"]
            self.ha[np.where(self.ha < 0)] += 2.0 * np.pi

            self.reward = np.zeros(self.observations.size, dtype=float)
            # Apply all the weights to the reward
            # In theory, could track where the reward is already
            # NaN and not bother doing extra computations.
            for key in self.weights:
                self.reward += self.weights[key] * getattr(self, key)(conditions)

            self.last_computed_reward = conditions.mjd
        return np.nanmax(self.reward)

    def generate_observations_rough(self, conditions):
        """Calculate reward function and highest reward observation.
        This is usually called by `generate_observations` which will
        take the result and then apply detailers to them.
        """
        max_reward = self.calc_reward_function(conditions)
        # take the first one in the array if there's a tie
        # Could change logic to return multiple pointings
        winner = np.min(np.where(self.reward == max_reward)[0])
        result = ObservationArray(n=1)
        result[0] = self.observations[winner].copy()
        return result

    def add_observation(self, observation, indx=None):
        """Let survey know about a completed observation."""
        # Check for a nore match
        indx = np.where(observation["scheduler_note"] == self.observations["scheduler_note"])[0]
        # Tracking arrays
        self.n_obs[indx] += 1
        self.last_observed[indx] = observation["mjd"]

        # If we are tracking n observations of some type:
        for key in self.tracking_notes:
            if key in observation["scheduler_note"]:
                self.tracking_notes[key] += 1

    def add_observations_array(self, observations_array_in, observations_hpid_in):
        """Like `add_observation`, but for a large array of completed
        observations."""
        for unote in np.unique(observations_array_in["scheduler_note"]):
            matching = np.where(observations_array_in["scheduler_note"] == unote)[0]
            indx = np.where(self.observations["scheduler_note"] == unote)[0]
            self.n_obs[indx] += np.size(matching)
            self.last_observed[indx] = observations_array_in["mjd"][matching].max()

            for key in self.tracking_notes:
                if key in unote:
                    self.tracking_notes[key] += np.size(matching)

    def ha_limit(self, conditions):
        """Apply hour angle limits."""
        result = self.zeros.copy()
        # apply hour angle limits
        result[
            np.where(
                (IntRounded(self.ha) > IntRounded(self.ha_max))
                & (IntRounded(self.ha) < IntRounded(self.ha_min))
            )
        ] = np.nan
        return result

    def alt_limit(self, conditions):
        """Apply altitude limits."""
        result = self.zeros.copy()
        result[np.where(IntRounded(self.alt) > IntRounded(self.alt_max))] = np.nan
        result[np.where(IntRounded(self.alt) < IntRounded(self.alt_min))] = np.nan
        return result

    def moon_limit(self, conditions):
        """Apply moon distanve limit."""
        result = self.zeros.copy()
        dists = _angular_separation(
            self.observations["RA"], self.observations["dec"], conditions.moon_ra, conditions.moon_dec
        )
        result[np.where(IntRounded(dists) < IntRounded(self.moon_dist_limit))] = np.nan
        return result

    def wind_limit(self, conditions):
        """Apply the wind limit."""
        result = self.zeros.copy()
        if conditions.wind_speed is None or conditions.wind_direction is None:
            return result
        wind_pressure = conditions.wind_speed * np.cos(self.az - conditions.wind_direction)
        result -= wind_pressure**2.0
        mask = IntRounded(wind_pressure) > IntRounded(self.wind_speed_maximum)
        result[mask] = np.nan

        return result

    def visit_gap(self, conditions):
        """Enforce a minimum visit gap."""
        diff = conditions.mjd - self.last_observed
        too_soon = np.where(IntRounded(diff) < IntRounded(self.gap_min))[0]
        result = self.zeros.copy()
        # Using NaN makes it a hard limit
        # could have a weight and just subtract from the reward
        result[too_soon] = np.nan
        return result

    def balance_revisit(self, conditions):
        """Code to balance revisiting different targets."""
        sum_obs = np.sum(self.n_obs)
        if sum_obs == 0:
            result = np.floor(1.0 + self.n_obs)
        else:
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

        # Could do a band check here and add a penalty for changing
        # bands

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
        keys of bandname and values of HEALpix arrays
        that are the darkest expected 5-sigma limiting depth
        expected for that region of sky
        """
        self.dark_map = {}
        for bandname in np.unique(self.observations["band"]):
            self.dark_map[bandname] = dark_m5(
                conditions.dec, bandname, conditions.site.latitude_rad, self.fiducial_FWHMEff
            )

    def m5diff(self, conditions):
        """Compute difference between current 5-sigma limiting depth and
        the depth at the same coordinates in "ideal" conditions.
        """
        if self.dark_map is None:
            self._dark_map(conditions)
        result = np.zeros(self.observations.size)
        for bandname in np.unique(self.observations["band"]):
            indx = np.where(self.observations["band"] == bandname)[0]
            diff_map = conditions.m5_depth[bandname] - self.dark_map[bandname]
            result[indx] = hp.get_interp_val(
                diff_map,
                np.degrees(self.observations["RA"][indx]),
                np.degrees(self.observations["dec"][indx]),
                lonlat=True,
            )

        return result

    def _reward_to_scalars(self, reward):
        scalar_reward = np.nanmax(reward)
        n_unmasked = np.sum(np.isfinite(reward))

        return scalar_reward, n_unmasked

    def make_reward_df(self, conditions, accum=True):
        """Create a pandas.DataFrame describing the reward from the
        survey."""

        feasibility = []
        max_rewards = []
        n_possibles = []
        accum_rewards = []
        accum_areas = []
        bf_label = []
        bf_class = []
        basis_weights = self.weights.values()

        short_labels = self.bf_short_labels

        tracking_reward = np.zeros(self.observations.size, dtype=float)

        for method_name in self.weights:
            bf_label.append(short_labels[method_name])
            bf_class.append(None)
            bf_reward = self.weights[method_name] * getattr(self, method_name)(conditions)
            max_reward, n_possible = self._reward_to_scalars(bf_reward)

            if np.isnan(max_reward) | (n_possible == 0):
                this_feasibility = False
            else:
                this_feasibility = True

            feasibility.append(this_feasibility)
            max_rewards.append(max_reward)
            n_possibles.append(n_possible)

            if accum:
                tracking_reward += bf_reward
                accum_reward, accum_area = self._reward_to_scalars(tracking_reward)
                accum_rewards.append(accum_reward)
                accum_areas.append(accum_area)

        reward_data = {
            "method": bf_label,
            "blank": bf_class,
            "feasible": feasibility,
            "max_reward": max_rewards,
            "n_possibles": n_possibles,
            "weight": basis_weights,
        }
        if accum:
            reward_data["max_accum_reward"] = accum_rewards
            reward_data["accum_n"] = accum_areas

        reward_df = pd.DataFrame(reward_data)

        return reward_df

    def reward_changes(self, conditions):
        """List the rewards for each basis function used by the survey."""

        reward_values = []

        for key in self.weights:
            reward_values.append(np.nanmax(getattr(self, key)(conditions)))

        names = list(self.weights.keys())

        return list(zip(names, reward_values))

    @property
    def bf_short_labels(self):
        long_labels = list(self.weights.keys())

        label_bases = [label.split(" @")[0] for label in long_labels]
        duplicated_labels = set([label for label in label_bases if label_bases.count(label) > 1])
        short_labels = []
        label_count = {k: 0 for k in duplicated_labels}
        for label_base in label_bases:
            if label_base in duplicated_labels:
                label_count[label_base] += 1
                short_labels.append(f"{label_base} {label_count[label_base]}")
            else:
                short_labels.append(label_base)

        label_map = dict(zip(long_labels, short_labels))

        return label_map
