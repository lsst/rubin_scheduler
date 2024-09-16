__all__ = (
    "BaseFeature",
    "BaseSurveyFeature",
    "NObsCount",
    "LastObservation",
    "NObservations",
    "LastObserved",
    "NObsNight",
    "PairInNight",
    "RotatorAngle",
    "NObservationsCurrentSeason",
    "LastNObsTimes",
    "NoteInNight",
    "NoteLastObserved",
)

import warnings

import healpy as hp
import numpy as np
from scipy.stats import binned_statistic

from rubin_scheduler.scheduler import utils
from rubin_scheduler.scheduler.utils import IntRounded
from rubin_scheduler.skybrightness_pre import dark_sky
from rubin_scheduler.utils import _hpid2_ra_dec, calc_season, survey_start_mjd


def send_unused_deprecation_warning(name):
    message = (
        f"The feature {name} is not in use by the current "
        "baseline scheduler and may be deprecated shortly. "
        "Please contact the rubin_scheduler maintainers if "
        "this is in use elsewhere."
    )
    warnings.warn(message, FutureWarning)


class BaseFeature:
    """The base class for features.
    This defines the standard API: a Feature should include
    a `self.feature` attribute, which could be a float, bool,
    or healpix size numpy array, or numpy masked array, and a
    __call__ method which returns `self.feature`.
    """

    def __init__(self, **kwargs):
        # self.feature should be a float, bool, or healpix size numpy
        # array, or numpy masked array
        self.feature = None

    def __call__(self):
        return self.feature


class BaseSurveyFeature(BaseFeature):
    """Track information relevant to the progress of a survey, using
    `self.feature` to hold this information.

    Features can track a single piece of information, or keep a map across
    the sky, or any other piece of information.

    Information in `self.feature` is updated via `add_observation` or
    `add_observation_array`.
    """

    def add_observations_array(self, observations_array, observations_hpid):
        """Update self.feature based on information in `observations_array`
        and `observations_hpid`.

        This is a method to more rapidly restore the feature to its
        expected state, by using an array of all previous observations
        (`observations_array`) instead looping over the individual
        observations, as in `self.add_observation`. The observations array
        allows rapid calculation of acceptable observations, without
        considering factors such as which healpix is relevant.

        The `observations_hpid` is a reorganized version of the
        `observations_array`, where the array representing each observation
        is extended to include `hpid` and recorded multiple times (for each
        healpixel). The `observations_hpid` allows rapid calculation of
        feature values which depend on hpid.

        Links between `observations_array` and `observations_hpid` can
        be made using `ID`.
        """
        print(self)
        raise NotImplementedError

    def add_observation(self, observation, indx=None, **kwargs):
        """Update self.feature based on information in `observation`.
        Observations should be ordered in monotonically increasing time.

        Parameters
        ----------
        observation : `np.array`, (1,N)
            Array of observation information, containing
            `mjd` for the time. See
            `rubin_scheduler.scheduler.utils.ObservationArray`.
        indx : `list`-like of [`int`]
            The healpixel indices that the observation overlaps.
            See `rubin_scheduler.utils.HpInLsstFov`.
        """
        raise NotImplementedError


class NoteInNight(BaseSurveyFeature):
    """Count appearances of any of `scheduler_notes` in
    observation `scheduler_note` in the
    current night; `note` must match one of `scheduler_notes`
    exactly.

    Useful for keeping track of how many times a survey or other subset
    of visits has executed in a given night.

    Parameters
    ----------
    notes : `list` [`str`], optional
        List of strings to match against observation `scheduler_note`
        values. The `scheduler_note` must match one of the items
        in notes exactly. Default of [None] will match any note.
    """

    def __init__(self, notes=[None]):
        self.feature = 0
        self.notes = notes
        self.current_night = -100

    def add_observations_array(self, observations_array, observations_hpid):
        # Identify the most recent night.
        if self.current_night != observations_array["night"][-1]:
            self.current_night = observations_array["night"][-1].copy()
            self.feature = 0
        # Identify observations within this night.
        indx = np.where(observations_array["night"] == observations_array["night"][-1])[0]
        # Count observations that match notes.
        for ind in indx:
            if (observations_array["scheduler_note"][ind] in self.notes) | (self.notes == [None]):
                self.feature += 1

    def add_observation(self, observation, indx=None):
        if self.current_night != observation["night"]:
            self.current_night = observation["night"].copy()
            self.feature = 0
        if (observation["scheduler_note"][0] in self.notes) | (self.notes == [None]):
            self.feature += 1


class NObsCount(BaseSurveyFeature):
    """Count the number of observations, whole sky (not per pixel).

    Because this feature will belong to a survey, it would count all
    observations that are counted for that survey.

    Parameters
    ----------
    note : `str` or None
        Count observations that match `str` in their scheduler_note field.
        Note can be a substring of scheduler_note, and will still match.
    filtername : `str` or None
        Optionally also (or independently) specify a filter to match.
    """

    def __init__(self, note=None, filtername=None):
        self.feature = 0
        self.note = note
        if self.note == "":
            self.note = None
        self.filtername = filtername

    def add_observations_array(self, observations_array, observations_hpid):
        if self.note is None and self.filtername is None:
            self.feature += observations_array.size
        elif self.note is None and self.filtername is not None:
            in_filt = np.where(observations_array["filter"] == self.filtername)
            self.feature += np.size(in_filt)
        elif self.note is not None and self.filtername is None:
            count = [self.note in note for note in observations_array["scheduler_note"]]
            self.feature += np.sum(count)
        else:
            # note and filtername are defined
            in_filt = np.where(observations_array["filter"] == self.filtername)
            count = [self.note in note for note in observations_array["scheduler_note"][in_filt]]
            self.feature += np.sum(count)

    def add_observation(self, observation, indx=None):
        # Track all observations
        if self.note is None and self.filtername is None:
            self.feature += 1
        elif self.note is None and self.filtername is not None:
            if observation["filter"][0] in self.filtername:
                self.feature += 1
        elif self.note is not None and self.filtername is None:
            if self.note in observation["scheduler_note"][0]:
                self.feature += 1
        else:
            if (observation["filter"][0] in self.filtername) and self.note in observation["scheduler_note"][
                0
            ]:
                self.feature += 1


class LastObservation(BaseSurveyFeature):
    """Track the last observation.
    Useful if you want to see when the
    last time a survey took an observation.

    Parameters
    ----------
    scheduler_note : `str` or None, optional
        Value of the scheduler_note to match, if not None.
    survey_name : `str` or None, optional
        Backwards compatible version of scheduler_note. Deprecated.
    """

    def __init__(self, scheduler_note=None, survey_name=None):
        if scheduler_note is None and survey_name is not None:
            self.scheduler_note = survey_name
        else:
            self.scheduler_note = scheduler_note
        # Start out with an empty observation
        self.feature = utils.ObservationArray()

    def add_observations_array(self, observations_array, observations_hpid):
        if self.scheduler_note is not None:
            valid_indx = np.ones(observations_array.size, dtype=bool)
            tmp = [self.scheduler_note in name for name in observations_array["scheduler_note"]]
            valid_indx = valid_indx * np.array(tmp)
            if len(tmp) > 0:
                self.feature = observations_array[valid_indx][-1]
        else:
            if len(observations_array) > 0:
                self.feature = observations_array[-1]

    def add_observation(self, observation, indx=None):
        if self.scheduler_note is not None:
            if self.scheduler_note in observation["scheduler_note"][0]:
                self.feature = observation
        else:
            self.feature = observation


class NObservations(BaseSurveyFeature):
    """
    Track the number of observations that have been made across the sky.

    Parameters
    ----------
    filtername : `str` or `list` [`str`] or None
        String or list that has all the filters that can count.
        Default None counts all filters.
    nside : `int`
        The nside of the healpixel map to use.
        Default None uses scheduler default.
    scheduler_note : `str` or None, optional
        The scheduler_note to match.
        Scheduler_note values which match this OR which contain this value
        as a subset of their string will match.
    survey_name : `str` or None
        The scheduler_note value to match.
        Deprecated in favor of scheduler_note, but provided for backward
        compatibility. Will be removed in the future.
    """

    def __init__(self, filtername=None, nside=None, scheduler_note=None, survey_name=None):
        if nside is None:
            nside = utils.set_default_nside()

        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)
        self.filtername = filtername
        if scheduler_note is None and survey_name is not None:
            self.scheduler_note = survey_name
        else:
            self.scheduler_note = scheduler_note
        # This feature is not used with "scheduler_note" in the baseline
        # survey. Should it really match scheduler_notes which contain
        # self.scheduler_note or should it be vice versa?
        # (i.e. this works as: self.scheduler_note = "survey" matches only
        # scheduler_note == "survey", but self.scheduler_note = "survey a"
        # will match both "survey" and "survey a".
        self.bins = np.arange(hp.nside2npix(nside) + 1) - 0.5

    def add_observations_array(self, observations_array, observations_hpid):
        valid_indx = np.ones(observations_hpid.size, dtype=bool)
        if self.filtername is not None:
            valid_indx[np.where(observations_hpid["filter"] != self.filtername)[0]] = False
        if self.scheduler_note is not None:
            tmp = [name in self.scheduler_note for name in observations_hpid["scheduler_note"]]
            valid_indx = valid_indx * np.array(tmp)
        data = observations_hpid[valid_indx]
        if np.size(data) > 0:
            result, _be, _bn = binned_statistic(
                data["hpid"], np.ones(data.size), statistic=np.sum, bins=self.bins
            )
            self.feature += result

    def add_observation(self, observation, indx=None):
        if self.filtername is None or observation["filter"][0] in self.filtername:
            if self.scheduler_note is None or observation["scheduler_note"][0] in self.scheduler_note:
                self.feature[indx] += 1


class LargestN:
    def __init__(self, n):
        # This is used within other features or basis functions,
        # but is not a feature itself
        self.n = n

    def __call__(self, in_arr):
        if np.size(in_arr) < self.n:
            return -1
        result = in_arr[-self.n]
        return result


class LastNObsTimes(BaseSurveyFeature):
    """Record the last three observations for each healpixel"""

    def __init__(self, filtername=None, n_obs=3, nside=None):
        self.filtername = filtername
        self.n_obs = n_obs
        if nside is None:
            nside = utils.set_default_nside()
        self.feature = np.zeros((n_obs, hp.nside2npix(nside)), dtype=float)
        self.bins = np.arange(hp.nside2npix(nside) + 1) - 0.5

    def add_observations_array(self, observations_array, observations_hpid):
        # Assumes we're already sorted on mjd
        valid_indx = np.ones(observations_hpid.size, dtype=bool)
        if self.filtername is not None:
            valid_indx[np.where(observations_hpid["filter"] != self.filtername)[0]] = False
        data = observations_hpid[valid_indx]

        if np.size(data) > 0:
            for i in range(1, self.n_obs + 1):
                func = LargestN(i)
                result, _be, _bn = binned_statistic(data["hpid"], data["mjd"], statistic=func, bins=self.bins)
                # some_vals = np.where(np.sum(result, axis=1) > 0)[0]
                self.feature[-i, :] = result

    def add_observation(self, observation, indx=None):
        if self.filtername is None or observation["filter"][0] in self.filtername:
            self.feature[0:-1, indx] = self.feature[1:, indx]
            self.feature[-1, indx] = observation["mjd"]


class NObservationsCurrentSeason(BaseSurveyFeature):
    """Count the observations at each healpix, taken in the most
    recent season, that meet filter, seeing and m5 criteria.

    Useful for ensuring that "good quality" observations are acquired
    in each season at each point in the survey footprint.

    Parameters
    ----------
    filtername : `str`, optional
        If None (default) count observations in any filter.
        Otherwise, only count observations in the specified filter.
    nside : `int`, optional
        If None (default), use default nside for scheduler.
        Otherwise, set nside for the map to nside.
    seeing_fwhm_max : `float`, optional
        If None (default), count observations up to any seeing value.
        Otherwise, only count observations with better seeing (`FWHMeff`)
        than `seeing_fwhm_max`. In arcseconds.
    m5_penalty_max : `float`, optional
        If None (default), count observations with any m5 value.
        Otherwise, only count observations within this value of the
        dark sky map at this pixel.
        Only relevant if filtername is not None.
    mjd_start : `float`, optional
        If None, uses default survey_start_mjd for the start of the survey.
        This defines the starting year for counting seasons, so should
        be the start of the survey.
    """

    def __init__(
        self,
        filtername=None,
        nside=None,
        seeing_fwhm_max=None,
        m5_penalty_max=None,
        mjd_start=None,
    ):
        if nside is None:
            nside = utils.set_default_nside()
        self.nside = nside
        self.seeing_fwhm_max = seeing_fwhm_max
        self.filtername = filtername
        self.m5_penalty_max = m5_penalty_max
        # If filtername not set, then we can't set m5_penalty_max.
        if self.filtername is None and self.m5_penalty_max is not None:
            warnings.warn("To use m5_penalty_max, filtername must be set. Disregarding m5_penalty_max.")
            self.m5_penalty_max = None

        # Get the dark sky map
        if self.filtername is not None:
            self.dark_map = dark_sky(nside)[filtername]
        self.ones = np.ones(hp.nside2npix(self.nside))

        if mjd_start is None:
            mjd_start = survey_start_mjd()
        self.mjd_start = mjd_start

        # Set up feature values - this includes the count in a given season
        # Find the healpixels for each point on the sky
        self.ra, self.dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))
        self.ra_deg = np.degrees(self.ra)

        self.mjd = mjd_start
        # Set up season_map. This should be the same as conditions.season_map
        # but the important thing is that mjd_start matches.
        self.season_map = calc_season(self.ra_deg, [self.mjd], self.mjd_start).flatten()
        self.season = np.floor(self.season_map)

        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)
        # Set bins for add_observations_array
        self.bins = np.arange(hp.nside2npix(nside) + 1) - 0.5

    def season_update(self, observation=None, conditions=None):
        """Update the season_map to the current time.

        This assumes time increases monotonically in the conditions or
        observations objects passed as arguments.
        Using the 'mjd' from the conditions, where parts of the sky
        have changed season (increasing in the `self.season_map` + 1)
        the `self.feature` is cleared, in order to restart counting
        observations in the new season.

        Parameters
        ----------
        observation : `np.array`, (1,N)
            Array of observation information, containing
            `mjd` for the time. See
            `rubin_scheduler.scheduler.utils.ObservationArray`.
        conditions : `rubin_scheduler.scheduler.Conditions`, optional
            A conditions object, containing `mjd`.

        Notes
        -----
        One of observations or conditions must be passed, but either
        are options so this can be used with add_observation.
        Most of the time, the updates will come from `conditions`.
        """
        mjd_now = None
        mjd_from = None
        if observation is not None:
            mjd_now = observation["mjd"]
            mjd_from = "observation"
        # Prefer to use Conditions for the time.
        if conditions is not None:
            mjd_now = conditions.mjd
            mjd_from = "conditions"
        if mjd_now is None:
            warnings.warn(
                "Expected either a conditions or observations object. Not updating season_map.",
                UserWarning,
            )
            return
        if isinstance(mjd_now, np.ndarray) or isinstance(mjd_now, list):
            mjd_now = mjd_now[0]
        # Check that time doesn't go backwards.
        # But only bother to warn if it's coming from conditions.
        # Observations may look like they go backwards.
        if mjd_now < self.mjd:
            if mjd_from == "conditions":
                warnings.warn(
                    f"Time must flow forwards to track the "
                    f"feature in {self.__class__.__name__}."
                    f"Not updating season_map.",
                    UserWarning,
                )
            # But just return without update in either case.
            return
        # Calculate updated season values.
        updated_season = np.floor(self.season_map + (mjd_now - self.mjd_start) / 365.25)
        # Clear feature where season increased by 1 (note 'floor' above).
        self.feature[np.where(updated_season > self.season)] = 0
        # Update to new time and season.
        self.mjd = mjd_now
        self.season = updated_season

    def add_observations_array(self, observations_array, observations_hpid):
        # Update self.season to the most recent observation time
        self.season_update(observation=observations_array[-1])

        # Start assuming 'check' is all True, for all observations.
        check = np.ones(observations_array.size, dtype=bool)

        # Rule out the observations with seeing fwhmeff > limit
        if self.seeing_fwhm_max is not None:
            check[np.where(observations_array["FWHMeff"] > self.seeing_fwhm_max)] = False

        # Rule out observations which are not in the desired filter
        if self.filtername is not None:
            check[np.where(observations_array["filter"] != self.filtername)] = False

        # Convert the "check" array on observations_array into a
        # "check" array on the hpids so we can evaluate healpix-level items
        check_hpid_indxs = np.in1d(observations_hpid["ID"], observations_array["ID"][check])
        # Set up a new valid/not valid flag, now on observations_hpid
        check_hp = np.zeros(len(observations_hpid), dtype=bool)
        # Set all observations which passed simpler tests above to True
        check_hp[check_hpid_indxs] = True

        hpids = observations_hpid["hpid"]

        # This could be done once per observation, but to make life
        # easier for index matching, let's just calculate season
        # based on the hpid array.
        # We only want to count observations that would fall within the
        # current season, so calculate the season for each obs.
        seasons = np.floor(
            calc_season(
                np.degrees(observations_hpid["RA"]),
                observations_hpid["mjd"],
                self.mjd_start,
                calc_diagonal=True,
            )
        )
        season_change = self.season[hpids] - seasons
        check_hp = np.where(season_change != 0, False, check_hp)
        # And check for m5_penalty_map
        if self.m5_penalty_max is not None:
            penalty = self.dark_map[hpids] - observations_hpid["fivesigmadepth"]
            check_hp = np.where(penalty > self.m5_penalty_max, False, check_hp)

        # Bin up the observations per hpid and add them to the feature.
        if np.sum(observations_hpid["hpid"][check_hp]) > 0:
            result, _be, _bn = binned_statistic(
                observations_hpid["hpid"][check_hp],
                observations_hpid["hpid"][check_hp],
                bins=self.bins,
                statistic=np.size,
            )
        else:
            result = 0
        # Add the resulting observations to feature.
        self.feature += result

    def add_observation(self, observation, indx):
        # Update the season map to the current time.
        self.season_update(observation=observation)
        # Check if *this* observation is in the current season
        # (This means we could add observations from the past, but
        # would count it against the current season).
        season_obs = np.floor(self.season_map[indx] + (observation["mjd"] - self.mjd_start) / 365.25)
        this_season_indx = np.array(indx)[np.where(season_obs == self.season[indx])]

        # Check if seeing is good enough.
        if self.seeing_fwhm_max is not None:
            check = observation["FWHMeff"] <= self.seeing_fwhm_max
        else:
            check = True

        # Check if observation is in the right filter
        if self.filtername is not None and check:
            if observation["filter"][0] != self.filtername:
                check = False
            else:
                # Check if observation in correct filter and deep enough.
                if self.m5_penalty_max is not None:
                    penalty = self.dark_map[this_season_indx] - observation["fivesigmadepth"]
                    this_season_indx = this_season_indx[np.where(penalty <= self.m5_penalty_max)]

        if check:
            self.feature[this_season_indx] += 1


class LastObserved(BaseSurveyFeature):
    """
    Track the MJD when a pixel was last observed.
    Assumes observations are added in chronological order.

    Parameters
    ----------
    filtername : `str` or None
        Track visits in a particular filter or any filter (None).
    nside : `int` or None
        Nside for the healpix map, default of None uses the scheduler default.
    fill : `float`
        Fill value to use where no observations have been found.
    """

    def __init__(self, filtername="r", nside=None, fill=np.nan):
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float) + fill
        self.bins = np.arange(hp.nside2npix(nside) + 1) - 0.5

    def add_observations_array(self, observations_array, observations_hpid):
        # Assumes we're already sorted on mjd
        valid_indx = np.ones(observations_hpid.size, dtype=bool)
        if self.filtername is not None:
            valid_indx[np.where(observations_hpid["filter"] != self.filtername)[0]] = False
        data = observations_hpid[valid_indx]

        if np.size(data) > 0:
            result, _be, _bn = binned_statistic(data["hpid"], data["mjd"], statistic=np.max, bins=self.bins)
            good = np.where(result > 0)
            self.feature[good] = result[good]

    def add_observation(self, observation, indx=None):
        if self.filtername is None:
            self.feature[indx] = observation["mjd"]
        elif observation["filter"][0] in self.filtername:
            self.feature[indx] = observation["mjd"]


class NoteLastObserved(BaseSurveyFeature):
    """Track the last time an observation with a particular `note` field was
    made.

    Parameters
    ----------
    note : `str`
        Substring to match an observation note field to keep track of.
    """

    def __init__(self, note, filtername=None):
        self.note = note
        self.filtername = filtername
        self.feature = None

    def add_observation(self, observation, indx=None):
        if self.note in observation["scheduler_note"][0] and (
            self.filtername is None or self.filtername == observation["filter"][0]
        ):
            self.feature = observation["mjd"]


class NObsNight(BaseSurveyFeature):
    """
    Track how many times a healpixel has been observed in a night.

    Parameters
    ----------
    filtername : `str` or None
        Filter to track. None tracks observations in any filter.
    nside : `int` or None
        Scale of the healpix map. Default of None uses the scheduler
        default nside.
    """

    def __init__(self, filtername="r", nside=None):
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=int)
        self.night = None

    def add_observation(self, observation, indx=None):
        if observation["night"] != self.night:
            self.feature *= 0
            self.night = observation["night"]
        if (self.filtername == "") | (self.filtername is None):
            self.feature[indx] += 1
        elif observation["filter"][0] in self.filtername:
            self.feature[indx] += 1


class PairInNight(BaseSurveyFeature):
    """
    Track how many pairs have been observed within a night at a given healpix.

    Parameters
    ----------
    gap_min : `float` (25.)
        The minimum time gap to consider a successful pair in minutes
    gap_max : `float` (45.)
        The maximum time gap to consider a successful pair (minutes)
    """

    def __init__(self, filtername="r", nside=None, gap_min=25.0, gap_max=45.0):
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)
        self.indx = np.arange(self.feature.size)
        self.last_observed = LastObserved(filtername=filtername)
        self.gap_min = IntRounded(gap_min / (24.0 * 60))  # Days
        self.gap_max = IntRounded(gap_max / (24.0 * 60))  # Days
        self.night = 0
        # Need to keep a full record of times and healpixels observed in
        # a night.
        self.mjd_log = []
        self.hpid_log = []

    def add_observations_array(self, observations_array, observations_hpid):
        # ok, let's just find the largest night and toss all those in one
        # at a time
        ## THIS IGNORES FILTER??
        most_recent_night = np.where(observations_hpid["night"] == np.max(observations_hpid["night"]))[0]
        obs_hpid = observations_hpid[most_recent_night]
        uid = np.unique(obs_hpid["ID"])
        for ind_id in uid:
            # maybe a faster searchsorted way to do this, but it'll work
            # for now
            good = np.where(obs_hpid["ID"] == ind_id)[0]
            self.add_observation(observations_hpid[good][0], observations_hpid[good]["hpid"])

    def add_observation(self, observation, indx=None):
        if self.filtername is None:
            infilt = True
        else:
            infilt = observation["filter"][0] in self.filtername
        if infilt:
            if indx is None:
                indx = self.indx
            # Clear values if on a new night
            if self.night != observation["night"]:
                self.feature *= 0.0
                self.night = observation["night"]
                self.mjd_log = []
                self.hpid_log = []

            # Look for the mjds that could possibly pair with observation
            mjd_diff = IntRounded(observation["mjd"] - np.array(self.mjd_log))
            # normally would use np.searchsorted, but need to use IntRounded
            # to be sure we are cross-platform repeatable.
            in_range_indx = np.where((mjd_diff > self.gap_min) & (mjd_diff < self.gap_max))[0]
            # Now check if any of the healpixels taken in the time gap
            # match the healpixels of the observation.
            if in_range_indx.size > 0:
                left = in_range_indx.min()
                right = in_range_indx.max() + 1
                matches = np.in1d(indx, self.hpid_log[left:right])
                self.feature[np.array(indx)[matches]] += 1

            # record the mjds and healpixels that were observed
            self.mjd_log.extend([np.max(observation["mjd"])] * np.size(indx))
            self.hpid_log.extend(list(indx))


class RotatorAngle(BaseSurveyFeature):
    """
    Track what rotation angles things are observed with.
    XXX-under construction
    """

    def __init__(self, filtername="r", binsize=10.0, nside=None):
        """"""
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        # Actually keep a histogram at each healpixel
        self.feature = np.zeros((hp.nside2npix(nside), 360.0 / binsize), dtype=float)
        self.bins = np.arange(0, 360 + binsize, binsize)

    def add_observation(self, observation, indx=None):
        if observation["filter"][0] == self.filtername:
            # I think this is how to broadcast things properly.
            self.feature[indx, :] += np.histogram(observation.rotSkyPos, bins=self.bins)[0]
