__all__ = (
    "BaseFeature",
    "BaseSurveyFeature",
    "NObsCount",
    "NObsSurvey",
    "LastObservation",
    "LastsequenceObservation",
    "LastFilterChange",
    "NObservations",
    "CoaddedDepth",
    "LastObserved",
    "NObsNight",
    "PairInNight",
    "RotatorAngle",
    "NObservationsSeason",
    "NObsCountSeason",
    "NObservationsCurrentSeason",
    "LastNObsTimes",
    "SurveyInNight",
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
from rubin_scheduler.utils import _hpid2_ra_dec, calc_season, m5_flat_sed, survey_start_mjd


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
            `rubin_scheduler.scheduler.utils.empty_observation`.
        indx : `list`-like of [`int`]
            The healpixel indices that the observation overlaps.
            See `rubin_scheduler.utils.HpInLsstFov`.
        """
        raise NotImplementedError


class SurveyInNight(BaseSurveyFeature):
    """Count appearances of `survey_str` within observation `note` in
    the current night; `survey_str` must be contained in `note`.

    Useful to keep track of how many times a survey has executed in a night.

    Parameters
    ----------
    survey_str : `str`, optional
        String to search for in observation `scheduler_note`.
        String does not have to match `scheduler_note` exactly,
        just be contained in `scheduler_note`.
        Default of "" means any observation will match.
    """

    def __init__(self, survey_str=""):
        self.feature = 0
        self.survey_str = survey_str
        self.night = -100
        send_unused_deprecation_warning(self.__class__.__name__)

    def add_observation(self, observation, indx=None):
        if observation["night"] != self.night:
            self.night = observation["night"]
            self.feature = 0

        if self.survey_str in observation["scheduler_note"]:
            self.feature += 1


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
        in notes exactly. Default of None uses `[]` and will match any note.
    """

    def __init__(self, notes=None):
        self.feature = 0
        if notes is None:
            notes = []
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
            if observations_array["scheduler_note"][ind] in self.notes:
                self.feature += 1

    def add_observation(self, observation, indx=None):
        if self.current_night != observation["night"]:
            self.current_night = observation["night"].copy()
            self.feature = 0
        if observation["scheduler_note"] in self.notes:
            self.feature += 1


class NObsCount(BaseSurveyFeature):
    """Count the number of observations.
    Total number, not tracked over sky

    Parameters
    ----------
    filtername : `str` (None)
        The filter to count (if None, all filters counted)

    """

    def __init__(self, filtername=None, tag=None):
        self.feature = 0
        self.filtername = filtername
        # 'tag' is used in GoalStrictFilterBasisFunction
        self.tag = tag
        if self.tag is not None:
            warnings.warn(
                "Tag is not a supported element"
                "of the `observation` and this aspect of "
                "the feature will be "
                "deprecated in 2 minor releases.",
                DeprecationWarning,
                stack_level=2,
            )

    def add_observations_array(self, observations_array, observations_hpid):
        if self.filtername is None:
            self.feature += np.size(observations_array)
        else:
            in_filt = np.where(observations_array["filter"] == self.filtername)[0]
            self.feature += np.size(in_filt)

    def add_observation(self, observation, indx=None):
        if (self.filtername is None) and (self.tag is None):
            # Track all observations
            self.feature += 1
        elif (
            (self.filtername is not None)
            and (self.tag is None)
            and (observation["filter"][0] in self.filtername)
        ):
            # Track all observations on a specified filter
            self.feature += 1
        elif (self.filtername is None) and (self.tag is not None) and (observation["tag"][0] in self.tag):
            # Track all observations on a specified tag
            self.feature += 1
        elif (
            (self.filtername is None)
            and (self.tag is not None)
            and
            # Track all observations on a specified filter on a specified tag
            (observation["filter"][0] in self.filtername)
            and (observation["tag"][0] in self.tag)
        ):
            self.feature += 1


class NObsCountSeason(BaseSurveyFeature):
    """Count the number of observations in a season.

    Parameters
    ----------
    filtername : `str` (None)
        The filter to count (if None, all filters counted)

    Notes
    -----
    Uses `season_calc` to calculate season value.

    Seems unused - added deprecation warning.
    """

    def __init__(
        self,
        season,
        nside=None,
        filtername=None,
        tag=None,
        season_modulo=2,
        offset=None,
        max_season=None,
        season_length=365.25,
    ):
        self.feature = 0
        self.filtername = filtername
        self.tag = tag
        self.season = season
        self.season_modulo = season_modulo
        if offset is None:
            self.offset = np.zeros(hp.nside2npix(nside), dtype=int)
        else:
            self.offset = offset
        self.max_season = max_season
        self.season_length = season_length
        send_unused_deprecation_warning(self.__class__.__name__)

    def add_observation(self, observation, indx=None):
        season = utils.season_calc(
            observation["night"],
            modulo=self.season_modulo,
            offset=self.offset[indx],
            max_season=self.max_season,
            season_length=self.season_length,
        )
        if self.season in season:
            if (self.filtername is None) and (self.tag is None):
                # Track all observations
                self.feature += 1
            elif (
                (self.filtername is not None)
                and (self.tag is None)
                and (observation["filter"][0] in self.filtername)
            ):
                # Track all observations on a specified filter
                self.feature += 1
            elif (self.filtername is None) and (self.tag is not None) and (observation["tag"][0] in self.tag):
                # Track all observations on a specified tag
                self.feature += 1
            elif (
                (self.filtername is None)
                and (self.tag is not None)
                and
                # Track all observations on a specified filter on a
                # specified tag
                (observation["filter"][0] in self.filtername)
                and (observation["tag"][0] in self.tag)
            ):
                self.feature += 1


class NObsSurvey(BaseSurveyFeature):
    """Count the number of observations, whole sky (not per pixel).

    Parameters
    ----------
    note : `str` (None)
        Only count observations that contain str in their note field
    """

    def __init__(self, note=None):
        self.feature = 0
        self.note = note

    def add_observation(self, observation, indx=None):
        # Track all observations
        if self.note is None:
            self.feature += 1
        else:
            if self.note in observation["scheduler_note"]:
                self.feature += 1


class LastObservation(BaseSurveyFeature):
    """Track the last observation.
    Useful if you want to see when the
    last time a survey took an observation.

    Parameters
    ----------
    survey_name : `str` (None)
        Only records if the survey name matches (or survey_name set to None)
    """

    def __init__(self, survey_name=None):
        # Will this work for observations read from a database???
        # "note" is definitely NOT guaranteed to match the survey_name.
        self.survey_name = survey_name
        # Start out with an empty observation
        self.feature = utils.empty_observation()

    def add_observations_array(self, observations_array, observations_hpid):
        if self.survey_name is not None:
            good = np.where(observations_array["scheduler_note"] == self.survey_name)[0]
            if np.size(good) < 0:
                self.feature = observations_array[good[-1]]
        else:
            if len(observations_array) > 0:
                self.feature = observations_array[-1]

    def add_observation(self, observation, indx=None):
        if self.survey_name is not None:
            if self.survey_name in observation["scheduler_note"]:
                self.feature = observation
        else:
            self.feature = observation


class LastsequenceObservation(BaseSurveyFeature):
    """When was the last observation"""

    def __init__(self, sequence_ids=""):
        self.sequence_ids = sequence_ids  # The ids of all sequence
        # observations...
        # Start out with an empty observation
        self.feature = utils.empty_observation()

    def add_observation(self, observation, indx=None):
        if observation["survey_id"] in self.sequence_ids:
            self.feature = observation


class LastFilterChange(BaseSurveyFeature):
    """Record when the filter last changed."""

    def __init__(self):
        self.feature = {"mjd": 0.0, "previous_filter": None, "current_filter": None}

    def add_observation(self, observation, indx=None):
        if self.feature["current_filter"] is None:
            self.feature["mjd"] = observation["mjd"][0]
            self.feature["previous_filter"] = None
            self.feature["current_filter"] = observation["filter"][0]
        elif observation["filter"][0] != self.feature["current_filter"]:
            self.feature["mjd"] = observation["mjd"][0]
            self.feature["previous_filter"] = self.feature["current_filter"]
            self.feature["current_filter"] = observation["filter"][0]


class NObservations(BaseSurveyFeature):
    """
    Track the number of observations that have been made across the sky.

    Parameters
    ----------
    filtername : `str` ('r')
        String or list that has all the filters that can count.
    nside : `int` (32)
        The nside of the healpixel map to use

    """

    def __init__(self, filtername=None, nside=None, survey_name=None):
        if nside is None:
            nside = utils.set_default_nside()

        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)
        self.filtername = filtername
        self.survey_name = survey_name
        self.bins = np.arange(hp.nside2npix(nside) + 1) - 0.5

    def add_observations_array(self, observations_array, observations_hpid):
        valid_indx = np.ones(observations_hpid.size, dtype=bool)
        if self.filtername is not None:
            valid_indx[np.where(observations_hpid["filter"] != self.filtername)[0]] = False
        if self.survey_name is not None:
            tmp = [name in self.survey_name for name in observations_hpid["scheduler_note"]]
            valid_indx = valid_indx * np.array(tmp)
        data = observations_hpid[valid_indx]
        if np.size(data) > 0:
            result, _be, _bn = binned_statistic(
                data["hpid"], np.ones(data.size), statistic=np.sum, bins=self.bins
            )
            self.feature += result

    def add_observation(self, observation, indx=None):
        if self.filtername is None or observation["filter"][0] in self.filtername:
            if self.survey_name is None or observation["scheduler_note"] in self.survey_name:
                self.feature[indx] += 1


class NObservationsSeason(BaseSurveyFeature):
    """
    Track the number of observations that have been made across sky

    Parameters
    ----------
    season : `int`
        Only count observations in this season (year).
    filtername : `str` ('r')
        String or list that has all the filters that can count.
    nside : `int` (32)
        The nside of the healpixel map to use
    offset : `int` (0)
        The offset to use when computing the season (days)
    modulo : `int` (None)
        How to mod the years when computing season

    Notes
    -----
    Uses `season_calc` to calculate season value.
    """

    def __init__(
        self,
        season,
        filtername=None,
        nside=None,
        offset=0,
        modulo=None,
        max_season=None,
        season_length=365.25,
    ):
        if offset is None:
            offset = np.zeros(hp.nside2npix(nside), dtype=int)
        if nside is None:
            nside = utils.set_default_nside()

        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)
        self.filtername = filtername
        ## How does this work if the default is 0 -- in add_observation
        # an index is referenced for offset, so the default should fail
        self.offset = offset
        self.modulo = modulo
        self.season = season
        self.max_season = max_season
        self.season_length = season_length
        send_unused_deprecation_warning(self.__class__.__name__)

    def add_observation(self, observation, indx=None):
        # How does this work if indx is None -- self.offset[indx] should fail
        observation_season = utils.season_calc(
            observation["night"],
            offset=self.offset[indx],
            modulo=self.modulo,
            max_season=self.max_season,
            season_length=self.season_length,
        )
        if self.season in observation_season:
            if self.filtername is None or observation["filter"][0] in self.filtername:
                self.feature[indx] += 1


class LargestN:
    def __init__(self, n):
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
            `rubin_scheduler.scheduler.utils.empty_observation`.
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
        result, _be, _bn = binned_statistic(
            observations_hpid["hpid"][check_hp],
            observations_hpid["hpid"][check_hp],
            bins=self.bins,
            statistic=np.size,
        )
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
            if observation["filter"] != self.filtername:
                check = False
            else:
                # Check if observation in correct filter and deep enough.
                if self.m5_penalty_max is not None:
                    penalty = self.dark_map[this_season_indx] - observation["fivesigmadepth"]
                    this_season_indx = this_season_indx[np.where(penalty <= self.m5_penalty_max)]

        if check:
            self.feature[this_season_indx] += 1


class CoaddedDepth(BaseSurveyFeature):
    """Track the co-added depth that has been reached across the sky

    Parameters
    ----------
    fwh_meff_limit : `float` (100)
        The effective FWHM of the seeing (arcsecond).
        Images will only be added to the coadded depth if the observation
        FWHM is less than or equal to the limit.  Default 100.
    """

    def __init__(self, filtername="r", nside=None, fwhm_eff_limit=100.0):
        if nside is None:
            nside = utils.set_default_nside()
        self.filtername = filtername
        self.fwhm_eff_limit = IntRounded(fwhm_eff_limit)
        # Starting at limiting mag of zero should be fine.
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)

    def add_observation(self, observation, indx=None):
        if observation["filter"] == self.filtername:
            if IntRounded(observation["FWHMeff"]) <= self.fwhm_eff_limit:
                m5 = m5_flat_sed(
                    observation["filter"],
                    observation["skybrightness"],
                    observation["FWHMeff"],
                    observation["exptime"],
                    observation["airmass"],
                )

                self.feature[indx] = 1.25 * np.log10(10.0 ** (0.8 * self.feature[indx]) + 10.0 ** (0.8 * m5))


class LastObserved(BaseSurveyFeature):
    """
    Track when a pixel was last observed.
    Assumes observations are added in chronological order.
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
        if self.note in observation["scheduler_note"] and (
            self.filtername is None or self.filtername == observation["filter"]
        ):
            self.feature = observation["mjd"]


class NObsNight(BaseSurveyFeature):
    """
    Track how many times something has been observed in a night
    (Note, even if there are two, it might not be a good pair.)

    Parameters
    ----------
    filtername : `str` ('r')
        Filter to track.
    nside : `int` (32)
        Scale of the healpix map

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
    Track how many pairs have been observed within a night

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
