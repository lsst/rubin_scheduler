__all__ = ["PointingsSurvey"]

import healpy as hp
import numpy as np

from rubin_scheduler.scheduler.detailers import ParallacticRotationDetailer
from rubin_scheduler.utils import _angular_separation, _approx_ra_dec2_alt_az, healbin

from .base_survey import BaseSurvey


class PointingsSurvey(BaseSurvey):
    """Survey object for managing a set list of potential pointings.

    Parameters
    ----------
    observations : np.array
        An array of observations, from e.g. rubin_scheduler.scheduler.utils.empty_observation
        expect "RA", "dec", and "note" to be filled, other columns ignored.
    gap_min : `float` (25.)
        The minimum gap to force between observations of the same spot (minutes)
    alt_min : `float`
        Altitude limit of the telescope (degrees). Default 20.
    alt_max : `float`
        Altitude limit of the telescope (degrees). Default 85.
    ha_max, ha_min : float (4,-4)
        hour angle limits (hours)
    weights : dict
        Dictionary with keys of method names and values of floats.
        Default of None uses {"visit_gap": 1.0, "balance_revisit": 1.0,
                              "wind_limit": 1.0, "slew_time": -1.0}
    cuts : list of str
        List with the method names to run which apply hard cuts to observations.
        Default of None uses ["ha_limit", "alt_limit", "moon_limit"]
    """

    def __init__(
        self,
        observations,
        gap_min=25.0,
        moon_dist_limit=30.0,
        weights=None,
        cuts=None,
        alt_max=85.0,
        alt_min=20.0,
        detailers=None,
        ha_max=4,
        ha_min=-4,
    ):
        # Not doing a super here, don't want to even have an nside defined.

        # Check that observations["note"] are unique, otherwise incoming
        # observations will get double-counted
        if np.size(np.unique(observations["note"])) != np.size(observations):
            raise ValueError("observations['note'] values are not unique")

        self.observations = observations
        self.gap_min = gap_min / 60.0 / 24.0  # to days
        self.moon_dist_limit = np.radians(moon_dist_limit)
        self.alt_max = np.radians(alt_max)
        self.alt_min = np.radians(alt_min)
        self.zeros = self.observations["RA"] * 0.0

        # convert hour angle limits to radians and 0-2pi
        self.ha_max = (np.radians(ha_max * 360 / 24) + 2 * np.pi) % (2 * np.pi)
        self.ha_min = (np.radians(ha_min * 360 / 24) + 2 * np.pi) % (2 * np.pi)

        # Arrays to track progress
        self.n_obs = np.zeros(self.zeros.size, dtype=int)
        self.last_observed = np.zeros(self.zeros.size, dtype=float)

        self.last_computed_reward = -1.0

        if weights is None:
            self.weights = {"visit_gap": 1.0, "balance_revisit": 1.0, "wind_limit": 1.0, "slew_time": -1.0}
        else:
            self.weights = weights

        if cuts is None:
            self.cuts = ["ha_limit", "alt_limit", "moon_limit"]
        else:
            self.cuts = cuts

        self.scheduled_obs = None
        # If there's no detailers, add one to set rotation to near zero
        if detailers is None:
            self.detailers = [ParallacticRotationDetailer()]
        else:
            self.detailers = detailers

    def _check_feasibility(self, conditions):
        result = True
        reward = self.calc_reward_function(conditions)
        if not np.isfinite(reward):
            result = False
        return result

    def calc_reward_function(self, conditions):
        #
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
            # go through and apply cuts
            for cut in self.cuts:
                self.reward += getattr(self, cut)(conditions)
            # Apply all the weights to the reward
            # In theory, could track where the reward is already
            # NaN and not bother doing extra computations.
            for key in self.weights:
                self.reward += self.weights[key] * getattr(self, key)(conditions)

            self.last_computed_reward = conditions.mjd
        return np.nanmax(self.reward)

    def generate_observations_rough(self, conditions):
        """ """
        max_reward = self.calc_reward_function(conditions)
        # take the first one in the array if there's a tie
        winner = np.min(np.where(self.reward == max_reward)[0])
        return [self.observations[winner].copy().reshape(1)]

    def add_observation(self, observation, indx=None):
        # Check if we have the same note. Maybe also check other things like exptime?
        indx = np.where(observation["note"] == self.observations["note"])[0]
        # Probably need to add a check that note is unique
        self.n_obs[indx] += 1
        self.last_observed[indx] = observation["mjd"]

    def add_observations_array(self, observations_array_in, observations_hpid_in):
        for unote in np.unique(observations_array_in["note"]):
            matching = np.where(observations_array_in["note"] == unote)[0]
            indx = np.where(self.observations["note"] == unote)[0]
            self.n_obs[indx] += np.size(matching)
            self.last_observed[indx] = observations_array_in["mjd"][matching].max()

    def ha_limit(self, conditions):
        result = self.zeros.copy()
        # apply hour angle limits
        result[np.where((self.ha > self.ha_max) & (self.ha < self.ha_min))] = np.nan
        return result

    def alt_limit(self, conditions):
        result = self.zeros.copy()
        result[np.where(self.alt > self.alt_max)] = np.nan
        result[np.where(self.alt < self.alt_min)] = np.nan
        return result

    def moon_limit(self, conditions):
        result = self.zeros.copy()
        dists = _angular_separation(
            self.observations["RA"], self.observations["dec"], conditions.moon_ra, conditions.moon_dec
        )
        result[np.where(dists < self.moon_dist_limit)] = np.nan
        return result

    def wind_limit(self, conditions):
        # Apply the wind limit
        result = self.zeros.copy()
        if conditions.wind_speed is None or conditions.wind_direction is None:
            return result
        wind_pressure = conditions.wind_speed * np.cos(self.az - conditions.wind_direction)
        result -= wind_pressure**2.0
        mask = wind_pressure > self.wind_speed_maximum
        result[mask] = np.nan

        return result

    def visit_gap(self, conditions):
        """Enforce a minimum visit gap"""
        diff = conditions.mjd - self.last_observed
        too_soon = np.where(diff < self.gap_min)[0]
        result = self.zeros.copy()
        # Using NaN makes it a hard limit
        # could have a weight and just subtract from the reward
        result[too_soon] = np.nan
        return result

    def balance_revisit(self, conditions):
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
        return result

    def viz(self, nside=128):
        # if we wanted to vizulize what's going on, we could do something like
        result = healbin(self.ra, self.dec, self.reward, nside=nside, reduceFunc=np.max)
        # or just plot color-coded points for each reward. Maybe empty circles at the NaN points
        return result
