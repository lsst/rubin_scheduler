__all__ = ("FieldSurvey",)

import copy
import warnings
from functools import cached_property

import numpy as np

from rubin_scheduler.utils import ra_dec2_hpid

from ..features import LastObservation, NObsCount
from ..utils import ObservationArray
from . import BaseSurvey


class FieldSurvey(BaseSurvey):
    """A survey class for running field surveys.

    Parameters
    ----------
    basis_functions : `list` [`rubin_scheduler.scheduler.basis_function`]
        List of basis_function objects
    detailers : `list` [`rubin_scheduler.scheduler.detailer`] objects
        The detailers to apply to the list of observations.
    RA : `float`
        The RA of the field (degrees)
    dec : `float`
        The dec of the field to observe (degrees)
    sequence : `list` [`str`]
        The sequence of observations to take. (specify which filters to use).
    nvisits : `dict` {`str`: `int`}
        Dictionary of the number of visits in each filter.
        Default of None will use a backup sequence of 20 visits per filter.
        Must contain all filters in sequence.
    exptime : `dict` {`str`: `float`}
        Dictionary of the exposure time for visits in each filter.
        Default of None will use a backup sequence of 38s in u, and
        29.2s in all other bands. Must contain all filters in sequence.
    nexp : dict` {`str`: `int`}
        Dictionary of the number of exposures per visit in each filter.
        Default of None will use a backup sequence of 1 exposure per visit
        in u band, 2 in all other bands. Must contain all filters in sequence.
    ignore_obs : `list` [`str`] or None
        Ignore observations with this string in the `scheduler_note`.
        Will ignore observations which match subsets of the string, as well as
        the entire string. Ignoring 'mysurvey23' will also ignore 'mysurvey2'.
    accept_obs : `list` [`str`] or None
        If match_obs is set, then ONLY observations which match these
        strings in the `scheduler_note` will be counted for the survey.
        A complete match must occur; substrings will not match.
        (for obs_array too??)
    survey_name : `str` or None.
        The name to give this survey, for debugging and visualization purposes.
        Also propagated to the 'target_name' in the observation.
        The default None will construct a name based on the
        RA/Dec of the field.
    scheduler_note : `str` or None
        The value to include in the scheduler note.
        The scheduler note is for internal, scheduler, use for the purposes of
        identifying observations to ignore or include for a survey or feature.
    readtime : `float`
        Readout time for computing approximate time of observing
        the sequence. (seconds)
    filter_change_time : `float`
        Filter change time, on average. Used for computing approximate
        time for the observing sequence. (seconds)
    nside : `float` or None
        Nside for computing survey basis functions and maps.
        The default of None will use rubin_scheduler.utils.set_default_nside().
    flush_pad : `float`
        How long to hold observations in the queue after they
        were expected to be completed (minutes).
    reward_value : `float`
        An unused kwarg, provided for backward compatibility.
    """

    def __init__(
        self,
        basis_functions,
        RA,
        dec,
        sequence="ugrizy",
        nvisits=None,
        exptimes=None,
        nexps=None,
        ignore_obs=None,
        accept_obs=None,
        survey_name=None,
        target_name=None,
        science_program=None,
        observation_reason=None,
        scheduler_note=None,
        readtime=2.4,
        filter_change_time=120.0,
        nside=None,
        flush_pad=30.0,
        detailers=None,
        reward_value=None,
        nexp=None,
    ):
        default_nvisits = {"u": 20, "g": 20, "r": 20, "i": 20, "z": 20, "y": 20}
        default_exptimes = {"u": 38, "g": 29.2, "r": 29.2, "i": 29.2, "z": 29.2, "y": 29.2}
        default_nexps = {"u": 1, "g": 2, "r": 2, "i": 2, "z": 2, "y": 2}

        # Deprecated kwarg messages
        if reward_value is not None:
            warnings.warn("reward_value has been unused and will be deprecated.")
        if nexp is not None:
            if nexps is not None:
                warnings.warn(
                    "Use one of `nexp` or `nexps`: `nexps` will be "
                    "supported going forward and will override."
                )
            if nexps is None:
                warnings.warn("Please use `nexps` in the future. " "Will adapt from `nexp` presently.")
                nexps = nexp

        self.ra = np.radians(RA)
        self.ra_hours = RA / 360.0 * 24.0
        self.dec = np.radians(dec)
        self.ra_deg, self.dec_deg = RA, dec

        self.survey_name = survey_name
        if self.survey_name is None:
            self._generate_survey_name(target_name=target_name)
        # Backfill target name if it wasn't set
        if target_name is None:
            target_name = self.survey_name

        super().__init__(
            nside=nside,
            basis_functions=basis_functions,
            detailers=detailers,
            ignore_obs=ignore_obs,
            survey_name=self.survey_name,
            target_name=target_name,
            science_program=science_program,
            observation_reason=observation_reason,
        )
        self.accept_obs = accept_obs
        if isinstance(self.accept_obs, str):
            self.accept_obs = [self.accept_obs]
        self.indx = ra_dec2_hpid(self.nside, self.ra_deg, self.dec_deg)

        # Set all basis function equal.
        self.basis_weights = np.ones(len(basis_functions)) / len(basis_functions)

        self.flush_pad = flush_pad / 60.0 / 24.0  # To days
        self.filter_sequence = []

        self.scheduler_note = scheduler_note
        if self.scheduler_note is None:
            self.scheduler_note = self.survey_name

        # This sets up what a requested "observation" looks like.
        # For sequences, each 'observation' is more than one exposure.
        # When generating actual observations, filters which are not available
        # are not included in the requested sequence.
        if nvisits is None:
            nvisits = default_nvisits
        if exptimes is None:
            exptimes = default_exptimes
        if nexps is None:
            nexps = default_nexps

        # Do a little shuffling if this was not configured quite as expected
        if isinstance(sequence, str):
            if isinstance(nvisits, (float, int)):
                nvisits = dict([(filtername, nvisits) for filtername in sequence])
            if isinstance(exptimes, (float, int)):
                exptimes = dict([(filtername, exptimes) for filtername in sequence])
            if isinstance(nexps, (float, int)):
                nexps = dict([(filtername, nexps) for filtername in sequence])

        if isinstance(sequence, str):
            self.observations = []
            for filtername in sequence:
                for j in range(nvisits[filtername]):
                    obs = ObservationArray()
                    obs["filter"] = filtername
                    obs["exptime"] = exptimes[filtername]
                    obs["RA"] = self.ra
                    obs["dec"] = self.dec
                    obs["nexp"] = nexps[filtername]
                    obs["scheduler_note"] = self.scheduler_note
                    self.observations.append(obs)
        else:
            self.observations = sequence

        # Let's just make this an array for ease of use
        self.observations = np.concatenate(self.observations)
        order = np.argsort(self.observations["filter"])
        self.observations = self.observations[order]

        n_filter_change = np.size(np.unique(self.observations["filter"]))

        # Make an estimate of how long a sequence will take.
        # Assumes no major rotational or spatial
        # dithering slowing things down.
        # Does not account for unavailable filters.
        self.approx_time = (
            np.sum(self.observations["exptime"] + readtime * self.observations["nexp"])
            + filter_change_time * n_filter_change
        )
        # convert to days, for internal approximation in timestep sizes
        self.approx_time /= 3600.0 / 24.0
        # This is the only index in the healpix arrays that will be considered
        self.indx = ra_dec2_hpid(self.nside, self.ra_deg, self.dec_deg)

        # Tucking this here so we can look at how many observations
        # recorded for this field and what was the last one.
        self.extra_features["ObsRecorded"] = NObsCount()
        self.extra_features["LastObs"] = LastObservation()

    def _generate_survey_name(self, target_name=None):
        if target_name is not None:
            self.survey_name = target_name
        else:
            self.survey_name = f"Field {self.ra_deg :.2f} {self.dec_deg :.2f}"

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

        XXX--TODO:  Need to decide if we want to develop check_continue,
        or instead hold the sequence in the survey, and be able to check
        it that way.
        (note that this may depend a lot on how the SchedulerCSC works)
        """
        return True

    def add_observation(self, observation, **kwargs):
        """Add observation one at a time."""
        # Check each possible ignore string
        checks = [io not in str(observation["scheduler_note"]) for io in self.ignore_obs]
        passed_ignore = all(checks)
        passed_accept = True
        if passed_ignore and self.accept_obs is not None:
            # Check if this observation matches any accept string.
            checks = [io == str(observation["scheduler_note"][0]) for io in self.accept_obs]
            passed_accept = any(checks)
        # I think here I have to assume observation is an
        # array and not a dict.
        if passed_ignore and passed_accept:
            for feature in self.extra_features:
                self.extra_features[feature].add_observation(observation, **kwargs)
            for bf in self.extra_basis_functions:
                self.extra_basis_functions[bf].add_observation(observation, **kwargs)
            for bf in self.basis_functions:
                bf.add_observation(observation, **kwargs)
            for detailer in self.detailers:
                detailer.add_observation(observation, **kwargs)
            self.reward_checked = False

    def add_observations_array(self, observations_array_in, observations_hpid_in):
        """Add an array of observations rather than one at a time

        Parameters
        ----------
        observations_array_in : ObservationArray
            An array of completed observations,
            rubin_scheduler.scheduler.utils.ObservationArray
        observations_hpid_in : np.array
            Same as observations_array_in, but larger and with an
            additional column for HEALpix id. Each observation is
            listed mulitple times, once for every HEALpix it overlaps.
        """
        # Just to be sure things are sorted
        observations_array_in.sort(order="mjd")
        observations_hpid_in.sort(order="mjd")

        # Copy so we don't prune things for other survey objects
        observations_array = observations_array_in.copy()
        observations_hpid = observations_hpid_in.copy()

        for ig in self.ignore_obs:
            not_ignore = np.where(np.char.find(observations_array["scheduler_note"], ig) == -1)[0]
            observations_array = observations_array[not_ignore]

            not_ignore = np.where(np.char.find(observations_hpid["scheduler_note"], ig) == -1)[0]
            observations_hpid = observations_hpid[not_ignore]

        if self.accept_obs is not None:
            accept_indx = []
            accept_hp_indx = []
            for acc in self.accept_obs:
                accept_indx.append(np.where(observations_array["scheduler_note"] == acc)[0])
                accept_hp_indx.append(np.where(observations_hpid["scheduler_note"] == acc)[0])
            accept = np.concatenate(accept_indx)
            accept_hp = np.concatenate(accept_hp_indx)
            observations_array = observations_array[accept]
            observations_hpid = observations_hpid[accept_hp]

        for feature in self.extra_features:
            self.extra_features[feature].add_observations_array(observations_array, observations_hpid)
        for bf in self.extra_basis_functions:
            self.extra_basis_functions[bf].add_observations_array(observations_array, observations_hpid)
        for bf in self.basis_functions:
            bf.add_observations_array(observations_array, observations_hpid)
        for detailer in self.detailers:
            detailer.add_observations_array(observations_array, observations_hpid)
        self.reward_checked = False

    def calc_reward_function(self, conditions):
        # only calculates reward at the index for the RA/Dec of the field
        self.reward_checked = True
        if self._check_feasibility(conditions):
            self.reward = 0
            for bf, weight in zip(self.basis_functions, self.basis_weights):
                basis_value = bf(conditions, indx=self.indx)
                self.reward += basis_value * weight

            if not np.isscalar(self.reward):
                self.reward = np.sum(self.reward[self.indx])

                if np.any(np.isinf(self.reward)):
                    self.reward = np.inf
        else:
            # If not feasible, negative infinity reward
            self.reward = -np.inf

        return self.reward

    def generate_observations_rough(self, conditions):
        result = []
        if self._check_feasibility(conditions):
            result = copy.deepcopy(self.observations)

            # Set the flush_by
            result["flush_by_mjd"] = conditions.mjd + self.approx_time + self.flush_pad

            # remove filters that are not mounted
            mask = np.isin(result["filter"], conditions.mounted_filters)
            result = result[mask]
            # Put current loaded filter first
            ind1 = np.where(result["filter"] == conditions.current_filter)[0]
            ind2 = np.where(result["filter"] != conditions.current_filter)[0]
            result = result[ind1.tolist() + (ind2.tolist())]

            # convert to list of array.
            final_result = [
                row.reshape(
                    1,
                )
                for row in result
            ]
            result = final_result

        return result

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} survey_name='{self.survey_name}'"
            f", RA={self.ra}, dec={self.dec} at {hex(id(self))}>"
        )
