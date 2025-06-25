__all__ = ("FieldSurvey", "FieldAltAzSurvey")

import copy
import warnings
from functools import cached_property

import numpy as np

from rubin_scheduler.utils import DEFAULT_NSIDE, _approx_alt_az2_ra_dec, _ra_dec2_hpid

from ..detailers import AltAz2RaDecDetailer
from ..features import LastObservation, NObsCount
from ..utils import ObservationArray, backfill_filter_from_band
from . import BaseSurvey


class FieldSurvey(BaseSurvey):
    """A survey class for running field surveys.

    Parameters
    ----------
    basis_functions : `list` [`rubin_scheduler.scheduler.basis_function`]
        List of basis_function objects.
    detailers : `list` [`rubin_scheduler.scheduler.detailer`] objects
        The detailers to apply to the list of observations.
    RA : `float`
        The RA of the field (degrees)
    dec : `float`
        The dec of the field to observe (degrees)
    sequence : `list` [`str`]
        The sequence of observations to take. (specify which bands to use).
    nvisits : `dict` {`str`: `int`}
        Dictionary of the number of visits in each band.
        Default of None will use a backup sequence of 20 visits per band.
        Must contain all bands in sequence.
    exptimes : `dict` {`str`: `float`}
        Dictionary of the exposure time for visits in each band.
        Default of None will use a backup sequence of 38s in u, and
        29.2s in all other bands. Must contain all bands in sequence.
    nexps : dict` {`str`: `int`}
        Dictionary of the number of exposures per visit in each band.
        Default of None will use a backup sequence of 1 exposure per visit
        in u band, 2 in all other bands. Must contain all bands in sequence.
    ignore_obs : `list` [`str`] or None
        Ignore observations with this string in the `scheduler_note`.
        Will ignore observations which match subsets of the string, as well as
        the entire string. Ignoring 'mysurvey23' will also ignore 'mysurvey2'.
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
    band_change_time : `float`
        Band change time, on average. Used for computing approximate
        time for the observing sequence. (seconds)
    nside : `float` or None
        Nside for computing survey basis functions and maps.
        The default of None will use rubin_scheduler.utils.set_default_nside().
    flush_pad : `float`
        How long to hold observations in the queue after they
        were expected to be completed (minutes).
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
        survey_name=None,
        target_name=None,
        science_program=None,
        observation_reason="field_target",
        scheduler_note=None,
        readtime=2.4,
        band_change_time=120.0,
        nside=DEFAULT_NSIDE,
        flush_pad=30.0,
        detailers=None,
        filter_change_time=None,
    ):
        default_nvisits = {"u": 20, "g": 20, "r": 20, "i": 20, "z": 20, "y": 20}
        default_exptimes = {"u": 38, "g": 29.2, "r": 29.2, "i": 29.2, "z": 29.2, "y": 29.2}
        default_nexps = {"u": 1, "g": 2, "r": 2, "i": 2, "z": 2, "y": 2}

        if filter_change_time is not None:
            warnings.warn("filter_change_time deprecated in favor of band_change_time", FutureWarning)
            band_change_time = filter_change_time

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
        # It's useful to save these as class attributes too
        self.target_name = target_name
        self.science_program = science_program
        self.observation_reason = observation_reason

        self.indx = _ra_dec2_hpid(self.nside, self.ra, self.dec)

        # Set all basis function equal.
        self.basis_weights = np.ones(len(basis_functions)) / len(basis_functions)

        self.flush_pad = flush_pad / 60.0 / 24.0  # To days
        self.band_sequence = []

        self.scheduler_note = scheduler_note
        if self.scheduler_note is None:
            self.scheduler_note = self.survey_name

        # This sets up what a requested "observation" looks like.
        # For sequences, each 'observation' is more than one exposure.
        # When generating actual observations, bands which are not available
        # are not included in the requested sequence.
        if nvisits is None:
            nvisits = default_nvisits
        if exptimes is None:
            exptimes = default_exptimes
        if nexps is None:
            nexps = default_nexps

        # Were we passed something like an ObservationArray or list[ObsArray]
        if isinstance(sequence, ObservationArray) | isinstance(sequence[0], ObservationArray):
            self.observations = sequence

        # Or was the sequence specified by filters, nvisits, exptimes, nexps
        else:
            # Do a little shuffling if we had the simplest config
            if isinstance(nvisits, (float, int)):
                nvisits = dict([(bandname, nvisits) for bandname in sequence])
            if isinstance(exptimes, (float, int)):
                exptimes = dict([(bandname, exptimes) for bandname in sequence])
            if isinstance(nexps, (float, int)):
                nexps = dict([(bandname, nexps) for bandname in sequence])

            self.observations = []
            for bandname in sequence:
                for j in range(nvisits[bandname]):
                    obs = ObservationArray()
                    obs["band"] = bandname
                    obs["exptime"] = exptimes[bandname]
                    obs["RA"] = self.ra
                    obs["dec"] = self.dec
                    obs["nexp"] = nexps[bandname]
                    obs["scheduler_note"] = self.scheduler_note
                    self.observations.append(obs)

        # Let's just make this an array for ease of use if not already
        if not isinstance(self.observations, ObservationArray):
            self.observations = np.concatenate(self.observations)
        # and backfill "band" if it was set by the user in "filter"
        self.observations = backfill_filter_from_band(self.observations)

        order = np.argsort(self.observations["band"])
        self.observations = self.observations[order]

        n_band_change = np.size(np.unique(self.observations["band"]))

        # Make an estimate of how long a sequence will take.
        # Assumes no major rotational or spatial
        # dithering slowing things down.
        # Does not account for unavailable bands.
        self.approx_time = (
            np.sum(self.observations["exptime"] + readtime * self.observations["nexp"])
            + band_change_time * n_band_change
        )
        # convert to days, for internal approximation in timestep sizes
        self.approx_time /= 3600.0 / 24.0
        # This is the only index in the healpix arrays that will be considered
        self.indx = _ra_dec2_hpid(self.nside, self.ra, self.dec)

        # Tucking this here so we can look at how many observations
        # recorded - both for any note and for this note.
        self.extra_features["ObsRecorded"] = NObsCount(scheduler_note=None)
        self.extra_features["LastObs"] = LastObservation(scheduler_note=None)
        self.extra_features["ObsRecorded_note"] = NObsCount(scheduler_note=self.scheduler_note)
        self.extra_features["LastObs_note"] = LastObservation(scheduler_note=self.scheduler_note)

    def _generate_survey_name(self, target_name=None):
        if target_name is not None:
            self.survey_name = target_name
        else:
            self.survey_name = f"Field {self.ra_deg :.2f} {self.dec_deg :.2f}"

    @cached_property
    def roi_hpid(self):
        hpid = _ra_dec2_hpid(self.nside, self.ra, self.dec)
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

    def calc_reward_function(self, conditions):
        # only calculates reward at the index for the RA/Dec of the field
        self.reward_checked = True
        if self._check_feasibility(conditions):
            self.reward = 0.0
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


class FieldAltAzSurvey(FieldSurvey):
    """A clone of FieldSurvey that takes alt,az rather than RA,dec.

    Parameters
    ----------
    basis_functions : `list` [`rubin_scheduler.scheduler.basis_function`]
        List of basis_function objects
    detailers : `list` [`rubin_scheduler.scheduler.detailer`] objects
        The detailers to apply to the list of observations.
    az : `float`
        The azimuth of the field (degrees)
    alt : `float`
        The altitude of the field to observe (degrees)
    sequence : `list` [`str`]
        The sequence of observations to take. (specify which bands to use).
    nvisits : `dict` {`str`: `int`}
        Dictionary of the number of visits in each band.
        Default of None will use a backup sequence of 20 visits per band.
        Must contain all bands in sequence.
    exptimes : `dict` {`str`: `float`}
        Dictionary of the exposure time for visits in each band.
        Default of None will use a backup sequence of 38s in u, and
        29.2s in all other bands. Must contain all bands in sequence.
    nexps : dict` {`str`: `int`}
        Dictionary of the number of exposures per visit in each band.
        Default of None will use a backup sequence of 1 exposure per visit
        in u band, 2 in all other bands. Must contain all bands in sequence.
    ignore_obs : `list` [`str`] or None
        Ignore observations with this string in the `scheduler_note`.
        Will ignore observations which match subsets of the string, as well as
        the entire string. Ignoring 'mysurvey23' will also ignore 'mysurvey2'.
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
    band_change_time : `float`
        Band change time, on average. Used for computing approximate
        time for the observing sequence. (seconds)
    nside : `float` or None
        Nside for computing survey basis functions and maps.
        The default of None will use rubin_scheduler.utils.set_default_nside().
    flush_pad : `float`
        How long to hold observations in the queue after they
        were expected to be completed (minutes)."""

    def __init__(
        self,
        basis_functions,
        az,
        alt,
        sequence="ugrizy",
        nvisits=None,
        exptimes=None,
        nexps=None,
        ignore_obs=None,
        survey_name=None,
        target_name=None,
        science_program=None,
        observation_reason="altaz_target",
        scheduler_note=None,
        readtime=2.4,
        band_change_time=120.0,
        nside=DEFAULT_NSIDE,
        flush_pad=30.0,
        detailers=None,
        filter_change_time=None,
    ):
        if filter_change_time is not None:
            warnings.warn("filter_change_time deprecated in favor of band_change_time", FutureWarning)
            band_change_time = filter_change_time

        if detailers is None:
            detailers = [AltAz2RaDecDetailer()]

        # Check that an AltAz detailer is present
        names_match = ["AltAz2RaDecDetailer" in det.__class__.__name__ for det in detailers]
        if not np.any(names_match):
            ValueError(
                "detailers list does not include a AltAz2RaDecDetailer which is needed for FieldAltAzSurvey"
            )

        self.az = np.radians(az)
        self.alt = np.radians(alt)

        self.alt_deg, self.az_deg = alt, az

        super().__init__(
            basis_functions=basis_functions,
            RA=0.0,
            dec=0.0,
            sequence=sequence,
            nvisits=nvisits,
            exptimes=exptimes,
            nexps=nexps,
            ignore_obs=ignore_obs,
            survey_name=survey_name,
            target_name=target_name,
            science_program=science_program,
            observation_reason=observation_reason,
            scheduler_note=scheduler_note,
            readtime=readtime,
            band_change_time=band_change_time,
            nside=nside,
            flush_pad=flush_pad,
            detailers=detailers,
        )
        # Clobber and set to None
        self.ra = None
        self.dec = None

        # Backfill target name if it wasn't set
        if target_name is None:
            target_name = self.survey_name

        # Default self.observations have ra,dec. Swap to alt,az
        self.observations["RA"] = None
        self.observations["dec"] = None
        self.observations["alt"] = self.alt
        self.observations["az"] = self.az

    @property
    def roi_hpid(self):
        ra, dec = _approx_alt_az2_ra_dec([self.alt], [self.az], self.lat, self.lon, self.mjd)
        hpid = _ra_dec2_hpid(self.nside, ra, dec)
        return hpid

    def _generate_survey_name(self, target_name=None):
        if target_name is not None:
            self.survey_name = target_name
        else:
            self.survey_name = f"Field alt,az {self.alt_deg :.2f} {self.az_deg :.2f}"

    def calc_reward_function(self, conditions):
        # Only calculates reward at the roi_hpid for the RA,dec coordinate
        # of the alt,az position at the current time.
        self.reward_checked = True
        self.mjd = conditions.mjd
        self.lat = np.radians(conditions.site.latitude)
        self.lon = np.radians(conditions.site.longitude)
        hpid = self.roi_hpid
        if self._check_feasibility(conditions):
            self.reward = 0.0
            for bf, weight in zip(self.basis_functions, self.basis_weights):
                basis_value = bf(conditions, indx=hpid)
                self.reward += basis_value * weight

            if not np.isscalar(self.reward):
                self.reward = np.sum(self.reward[hpid])

                if np.any(np.isinf(self.reward)):
                    self.reward = np.inf
        else:
            # If not feasible, negative infinity reward
            self.reward = -np.inf

        return self.reward
