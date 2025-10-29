__all__ = ("DeepDrillingSurvey", "dd_bfs")

import copy
import logging
import random
from functools import cached_property

import numpy as np

import rubin_scheduler.scheduler.basis_functions as basis_functions
from rubin_scheduler.scheduler import features
from rubin_scheduler.scheduler.surveys import BaseSurvey
from rubin_scheduler.scheduler.utils import ObservationArray
from rubin_scheduler.utils import DEFAULT_NSIDE, ra_dec2_hpid

log = logging.getLogger(__name__)


class DeepDrillingSurvey(BaseSurvey):
    """A survey class for running deep drilling fields.

    Parameters
    ----------
    basis_functions : list of rubin_scheduler.scheduler.basis_function
        These should be feasibility basis functions.
    RA : float
        The RA of the field (degrees)
    dec : float
        The dec of the field to observe (degrees)
    sequence : list of observation objects or str (rgizy)
        The sequence of observations to take. Can be a string of
        list of obs objects.
    nvis : list of ints
        The number of visits in each band. Should be same length
        as sequence.
    survey_name : str (DD)
        The name to give this survey so it can be tracked
    reward_value : float (101.)
        The reward value to report if it is able to start (unitless).
    readtime : float (2.)
        Readout time for computing approximate time of observing
        the sequence. (seconds)
    flush_pad : float (30.)
        How long to hold observations in the queue after they
        were expected to be completed (minutes).
    """

    def __init__(
        self,
        basis_functions,
        RA,
        dec,
        sequence="rgizy",
        nvis=[20, 10, 20, 26, 20],
        exptime=30.0,
        u_exptime=30.0,
        nexp=2,
        ignore_obs=None,
        survey_name="DD",
        reward_value=None,
        readtime=2.0,
        band_change_time=120.0,
        nside=DEFAULT_NSIDE,
        flush_pad=30.0,
        seed=42,
        detailers=None,
        science_program=None,
    ):
        super(DeepDrillingSurvey, self).__init__(
            nside=nside,
            basis_functions=basis_functions,
            detailers=detailers,
            ignore_obs=ignore_obs,
            science_program=science_program,
        )
        random.seed(a=seed)

        self.ra = np.radians(RA)
        self.ra_hours = RA / 360.0 * 24.0
        self.dec = np.radians(dec)
        self.survey_name = survey_name
        self.reward_value = reward_value
        self.flush_pad = flush_pad / 60.0 / 24.0  # To days
        self.band_sequence = []
        if isinstance(sequence, str):
            self.observations = []
            for num, bandname in zip(nvis, sequence):
                for j in range(num):
                    obs = ObservationArray()
                    obs["band"] = bandname
                    if bandname == "u":
                        obs["exptime"] = u_exptime
                    else:
                        obs["exptime"] = exptime
                    obs["RA"] = self.ra
                    obs["dec"] = self.dec
                    obs["nexp"] = nexp
                    obs["scheduler_note"] = survey_name
                    self.observations.append(obs)
        else:
            self.observations = sequence

        # Let's just make this an array for ease of use
        self.observations = np.concatenate(self.observations)
        order = np.argsort(self.observations["band"])
        self.observations = self.observations[order]

        n_band_change = np.size(np.unique(self.observations["band"]))

        # Make an estimate of how long a seqeunce will take.
        # Assumes no major rotational or spatial
        # dithering slowing things down.
        self.approx_time = (
            np.sum(self.observations["exptime"] + readtime * self.observations["nexp"]) / 3600.0 / 24.0
            + band_change_time * n_band_change / 3600.0 / 24.0
        )  # to days

        if self.reward_value is None:
            self.extra_features["Ntot"] = features.NObsCount()
            self.extra_features["N_survey"] = features.NObsCount(note=self.survey_name)

    @cached_property
    def roi_hpid(self):
        hpid = ra_dec2_hpid(self.nside, np.degrees(self.ra), np.degrees(self.dec))
        return hpid

    def check_continue(self, observation, conditions):
        # feasibility basis functions?
        """
        This method enables external calls to check if a given
        observations that belongs to this survey is
        feasible or not. This is called once a sequence has
        started to make sure it can continue.

        XXX--TODO:  Need to decide if we want to develope check_continue,
        or instead hold the sequence in the survey, and be able to check
        it that way.
        """

        result = True

        return result

    def calc_reward_function(self, conditions):
        result = -np.inf
        if self._check_feasibility(conditions):
            if self.reward_value is not None:
                result = self.reward_value
            else:
                # XXX This might backfire if we want to have DDFs with
                # different fractions of the survey time. Then might need
                # to define a goal fraction, and have the reward be the
                # number of observations behind that target fraction.
                result = self.extra_features["Ntot"].feature / (self.extra_features["N_survey"].feature + 1)
        return result

    def generate_observations_rough(self, conditions):
        result = []
        if self._check_feasibility(conditions):
            result = copy.deepcopy(self.observations)

            # Set the flush_by
            result["flush_by_mjd"] = conditions.mjd + self.approx_time + self.flush_pad

            # remove bands that are not mounted
            mask = np.isin(result["band"], conditions.mounted_bands)
            result = result[mask]
            # Put current loaded band first
            ind1 = np.where(result["band"] == conditions.current_band)[0]
            ind2 = np.where(result["band"] != conditions.current_band)[0]
            result = result[ind1.tolist() + (ind2.tolist())]

        return result

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} survey_name='{self.survey_name}'"
            f", RA={self.ra}, dec={self.dec} at {hex(id(self))}>"
        )


def dd_bfs(
    RA,
    dec,
    survey_name,
    ha_limits,
    frac_total=0.0185 / 2.0,
    aggressive_frac=0.011 / 2.0,
    delays=[0.0, 0.5, 1.5],
    time_needed=62.0,
    wind_speed_maximum=20.0,
    nside=None,
):
    """
    Convienence function to generate all the feasibility basis functions
    """
    sun_alt_limit = -18.0
    fractions = [0.00, aggressive_frac, frac_total]
    bfs = []
    bfs.append(basis_functions.NotTwilightBasisFunction(sun_alt_limit=sun_alt_limit))
    bfs.append(basis_functions.TimeToTwilightBasisFunction(time_needed=time_needed))
    bfs.append(basis_functions.AvoidDirectWind(wind_speed_maximum=wind_speed_maximum, nside=nside))
    bfs.append(basis_functions.HourAngleLimitBasisFunction(RA=RA, ha_limits=ha_limits))
    bfs.append(basis_functions.MoonDownBasisFunction())
    bfs.append(basis_functions.FractionOfObsBasisFunction(frac_total=frac_total, survey_name=survey_name))
    bfs.append(
        basis_functions.LookAheadDdfBasisFunction(
            frac_total,
            aggressive_frac,
            sun_alt_limit=sun_alt_limit,
            time_needed=time_needed,
            RA=RA,
            survey_name=survey_name,
            ha_limits=ha_limits,
        )
    )
    bfs.append(
        basis_functions.SoftDelayBasisFunction(fractions=fractions, delays=delays, survey_name=survey_name)
    )
    bfs.append(basis_functions.TimeToScheduledBasisFunction(time_needed=time_needed))

    return bfs
