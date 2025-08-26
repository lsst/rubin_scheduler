__all__ = ("CoreScheduler",)

import logging
import time
from collections import OrderedDict
from copy import deepcopy
from io import StringIO

import healpy as hp
import numpy as np
import pandas as pd
from astropy.time import Time

from rubin_scheduler.scheduler.utils import HpInComcamFov, HpInLsstFov, IntRounded, ObservationArray
from rubin_scheduler.utils import DEFAULT_NSIDE, SURVEY_START_MJD, _hpid2_ra_dec, rotation_converter


class CoreScheduler:
    """Core scheduler that takes requests observations,
    reports observatory status and provides completed observations.

    Parameters
    ----------
    surveys : list (or list of lists) of rubin_scheduler.scheduler.survey
        A list of surveys to consider.
        If multiple surveys return the same highest reward value,
        the survey at the earliest position in the list will be selected.
        Can also be a list of lists to make heirarchical priorities
        ('tiers').
    nside : `int`
        A HEALpix nside value.
    camera : `str`
        Which camera to use to compute the correspondence between
        visits and HEALpixels when recording observations.
        Can be 'LSST' or 'comcam'.
    log : `logging.Logger`
        If None, a new logger is created.
    keep_rewards : `bool`
        Flag whether to record the rewards and basis function values
        (this can be useful for schedview).
    telescope : `str`
        Which telescope model to use for rotTelPos/rotSkyPos conversions.
        Default "rubin".
    target_id_counter : `int`
        Starting value for the target_id. If restarting observations, could
        be useful to set to whatever value the scheduler was at previously.
        Default 0.
    band2filter : `dict`
        Dict for mapping band name to filter name if filter has not been
        specified. Default of None maps 'ugrizy' to the same. Empty dict
        will do no mapping for missing "filter" values.
    survey_start_mjd : `float`
            The starting MJD of the survey.
    """

    def __init__(
        self,
        surveys,
        nside=DEFAULT_NSIDE,
        camera="LSST",
        log=None,
        keep_rewards=False,
        telescope="rubin",
        target_id_counter=0,
        band_to_filter=None,
        survey_start_mjd=SURVEY_START_MJD,
    ):
        self.keep_rewards = keep_rewards
        self.survey_start_mjd = survey_start_mjd
        # Use integer ns just to be sure there are no rounding issues.
        self.mjd_perf_counter_offset = np.int64(Time.now().mjd * 86400000000000) - time.perf_counter_ns()

        if log is None:
            self.log = logging.getLogger(type(self).__name__)
        else:
            self.log = log.getChild(type(self).__name__)

        # initialize a queue of observations to request
        self.queue = []
        # The indices of self.survey_lists that provided
        # the last addition(s) to the queue
        self.survey_index = [None, None]

        # If we have a list of survey objects, convert to list-of-lists
        if isinstance(surveys[0], list):
            self.survey_lists = surveys
        else:
            self.survey_lists = [surveys]
        self.nside = nside
        hpid = np.arange(hp.nside2npix(nside))
        self.ra_grid_rad, self.dec_grid_rad = _hpid2_ra_dec(nside, hpid)
        # Should just make camera a class that takes a pointing
        # and returns healpix indices
        if camera == "LSST":
            self.pointing2hpindx = HpInLsstFov(nside=nside)
        elif camera == "comcam":
            self.pointing2hpindx = HpInComcamFov(nside=nside)
        else:
            raise ValueError("camera %s not implamented" % camera)

        self.rc = rotation_converter(telescope=telescope)

        # Keep track of how many observations get flushed from the queue
        self.flushed = 0

        # Counter for observations added to the queue
        self.target_id_counter = target_id_counter

        # Set to something so it doesn't fail if never set later
        self.queue_fill_mjd_ns = -1
        self.queue_reward_df = None

        # Set mapping of band to filter if filter is missing
        if band_to_filter is None:
            self.band_to_filter_dict = {}
            for bandname in "ugrizy":
                self.band_to_filter_dict[bandname] = bandname
        else:
            self.band_to_filter_dict = band_to_filter

    def set_target_id_counter(self, target_id_counter):
        """In case can't be set on init.

        Parameters
        ----------
        target_id_counter : `int`
            What to start counting targets at.
        """
        self.target_id_counter = target_id_counter

    def set_band_to_filter(self, band_to_filter):
        """In case can't be set on init.

        Parameters
        ----------
        band_to_filter : `dict`
            Dictionary to use to map band names (ugrizy) to
            physical filter names.
        """
        self.band_to_filter_dict = band_to_filter

    def flush_queue(self):
        """Like it sounds, clear any currently queued desired observations."""
        self.queue = []
        self.survey_index = [None, None]

    def add_observations_array(self, obs):
        """Like add_observation, but for passing many observations at once.

        Assigns overlapping HEALpix IDs to each observation, then passes
        the observation array and constructed observations + healpix id
        to each survey.
        """
        # Need to add "band" here if it wasn't populated
        missing_band = np.where(obs["band"] == "")
        band = np.char.rstrip(obs["filter"][missing_band], chars="_0123456789")
        obs["band"][missing_band] = band

        obs.sort(order="mjd")
        # Generate list-of-lists for HEALPix IDs for each pointing
        lol_hpids = self.pointing2hpindx(obs["RA"], obs["dec"])
        hpids = []
        big_array_indx = []
        # Unravel the list-of-lists
        for i, indxs in enumerate(lol_hpids):
            for indx in indxs:
                hpids.append(indx)
                big_array_indx.append(i)
        hpids = np.array(hpids, dtype=[("hpid", int)])
        names = list(obs.dtype.names)
        types = [obs[name].dtype for name in names]

        names.append(hpids.dtype.names[0])
        types.append(hpids["hpid"].dtype)
        ndt = list(zip(names, types))

        obs_array_hpid = np.empty(hpids.size, dtype=ndt)
        obs_array_hpid[list(obs.dtype.names)] = obs[big_array_indx]
        obs_array_hpid[hpids.dtype.names[0]] = hpids

        for surveys in self.survey_lists:
            for survey in surveys:
                survey.add_observations_array(obs, obs_array_hpid)

        if np.max(obs["target_id"]) >= self.target_id_counter:
            self.target_id_counter = np.max(obs["target_id"]) + 1

    def add_observation(self, observation):
        """
        Record a completed observation and update features accordingly.

        Parameters
        ----------
        observation : dict-like
            An object that contains the relevant information about a
            completed observation
            (e.g., mjd, ra, dec, filter, rotation angle, etc)
        """
        # Catch if someone passed in a slice of an observation
        # rather than a full observation array
        if len(observation.shape) == 0:
            full_obs = ObservationArray()
            full_obs[0] = observation
            observation = full_obs

        # Need to add "band" here if it wasn't populated
        missing_band = np.where(observation["band"] == "")
        band = np.char.rstrip(observation["filter"][missing_band], chars="_0123456789")
        observation["band"][missing_band] = band

        # Find the healpixel centers that are included in an observation
        indx = self.pointing2hpindx(
            observation["RA"][0], observation["dec"][0], rotSkyPos=observation["rotSkyPos"][0]
        )
        for surveys in self.survey_lists:
            for survey in surveys:
                survey.add_observation(observation, indx=indx)

        if np.max(observation["target_id"]) >= self.target_id_counter:
            self.target_id_counter = np.max(observation["target_id"]) + 1

    def update_conditions(self, conditions_in):
        """
        Parameters
        ----------
        conditions : dict-like
            The current conditions of the telescope (pointing position,
            loaded filters, cloud-mask, etc)
        """
        # Add the current queue and scheduled queue to the conditions
        self.conditions = conditions_in

        # Use our own def of survey start
        self.conditions.survey_start_mjd = self.survey_start_mjd

        # put the local queue in the conditions
        self.conditions.queue = self.queue

        # Update conditions in all survey objects so they
        # can generate scheduled observations for any new ToO events
        for surveys in self.survey_lists:
            for survey in surveys:
                survey.update_conditions(conditions_in)

        # Check if any surveys have upcomming scheduled observations.
        # Note that we are accumulating all of the possible scheduled
        # observations, so it's up to the user to make sure things don't
        # collide. The ideal implementation would be to have all the
        # scheduled observations in a single survey objects,
        # presumably at the highest tier of priority.

        all_scheduled = []
        for sl in self.survey_lists:
            for sur in sl:
                scheduled = sur.get_scheduled_obs()
                if scheduled is not None:
                    all_scheduled.append(scheduled.view(np.ndarray))
        if len(all_scheduled) == 0:
            self.conditions.scheduled_observations = []
        else:
            all_scheduled = np.sort(np.concatenate(all_scheduled).ravel())
            # In case the surveys have not been removing executed observations
            all_scheduled = all_scheduled[np.where(all_scheduled >= self.conditions.mjd)]
            self.conditions.scheduled_observations = all_scheduled

    def _check_queue_mjd_only(self, mjd):
        """
        Check if there are things in the queue that can be executed
        using only MJD and not full conditions.
        This is primarily used by sim_runner to reduce calls calculating
        updated conditions when they are not needed.
        """
        result = False
        if len(self.queue) > 0:
            if (IntRounded(mjd) < IntRounded(self.queue[0]["flush_by_mjd"])) | (
                self.queue[0]["flush_by_mjd"] == 0
            ):
                result = True
        return result

    def request_observation(self, mjd=None, whole_queue=False):
        """
        Ask the scheduler what it wants to observe next

        Parameters
        ----------
        mjd : `float`
            The Modified Julian Date.
            If None, it uses the MJD from the conditions from the
            last conditions update.
        whole_queue : `bool`
            Return the entire list of observations in the queue (True),
            or just a single observation (False). Default False

        Returns
        -------
        observation : `list` [`~.scheduler.utils.ObservationArray`]
            Returns ~.scheduler.utils.ObservationArray if
            whole_queue is False
            Returns None if the queue fails to fill
            Returns list of ObservationArray if whole_queue is True

        Notes
        -----
        Calling request_observation repeatedly, even without updating
        the time or conditions, can return different requested
        observations. This is because the internal queue is filled
        when it becomes empty, and then subsequent calls to
        request_observation will pop successive visits from this queue,
        until the queue is empty or is flushed.
        """
        if mjd is None:
            mjd = self.conditions.mjd
        if len(self.queue) == 0:
            self._fill_queue()

        if len(self.queue) == 0:
            return None
        else:
            # If the queue has gone stale, flush and refill.
            # Zero means no flush_by was set.
            if (IntRounded(mjd) > IntRounded(self.queue[0]["flush_by_mjd"])) & (
                self.queue[0]["flush_by_mjd"] != 0
            ):
                self.flushed += len(self.queue)
                self.flush_queue()
                self._fill_queue()
            if len(self.queue) == 0:
                return None
            if whole_queue:
                result = deepcopy(self.queue)
                self.flush_queue()
            else:
                result = self.queue.pop(0)

            # TODO : Remove this hack which is for use with ts_scheduler
            # version <=v2.3 .. remove ts_scheduler actually drops "note".
            for obs in result:
                obs["note"] = obs["scheduler_note"]
            return result

    def _fill_queue(self):
        """
        Compute reward function for each survey and fill the observing queue
        with the observations from the highest reward survey.
        """

        # If we loaded the scheduler from a pickle, self.keep_rewards
        # might not have been initialized, but we want it to succeed
        # anyway. So, assign it to false if it isn't present.
        try:
            keep_rewards = self.keep_rewards
        except AttributeError:
            keep_rewards = False

        if keep_rewards:
            # Use perf_counter_ns to get the best time resolution to guarantee
            # successive calls do occur within the same resolution element.
            self.queue_fill_mjd_ns = np.int64(self.mjd_perf_counter_offset + time.perf_counter_ns())
            self.queue_reward_df = self.make_reward_df(accum=False)
            self.queue_reward_df = self.queue_reward_df.assign(
                queue_start_mjd=np.max(self.conditions.mjd),
                queue_fill_mjd_ns=self.queue_fill_mjd_ns,
            )

        rewards = None
        for ns, surveys in enumerate(self.survey_lists):
            rewards = np.zeros(len(surveys))
            for i, survey in enumerate(surveys):
                # For each survey, find the highest reward value.
                all_rewards_this_survey = survey.calc_reward_function(self.conditions)
                rewards[i] = (
                    np.nan
                    if np.all(np.isnan(all_rewards_this_survey))
                    else np.nanmax(all_rewards_this_survey)
                )
            # If we have a tier with a good reward, break out of the loop
            if np.nanmax(rewards) > -np.inf:
                self.survey_index[0] = ns
                break
        if (np.nanmax(rewards) == -np.inf) | (np.isnan(np.nanmax(rewards))):
            self.flush_queue()
        else:
            to_fix = np.where(np.isnan(rewards))
            rewards[to_fix] = -np.inf
            # Take a min here, so the surveys will be executed in the order
            # they are entered if there is a tie.
            self.survey_index[1] = np.min(np.where(rewards == np.nanmax(rewards)))
            # Survey returns ObservationArray
            result = self.survey_lists[self.survey_index[0]][self.survey_index[1]].generate_observations(
                self.conditions
            )
            # Tag with a unique target_id
            result["target_id"] = np.arange(self.target_id_counter, self.target_id_counter + result.size)
            self.target_id_counter += result.size

            # Check that filter has been set
            need_filtername_indx = np.where((result["filter"] == "") | (result["filter"] is None))[0]
            if len(self.band_to_filter_dict) > 0:
                for indx in need_filtername_indx:
                    result[indx]["filter"] = self.band_to_filter_dict[result[indx]["band"]]

            # Convert to a list for the queue
            self.queue = result.tolist()
            self.queue_filled = self.conditions.mjd

        if len(self.queue) == 0:
            self.log.warning(f"Failed to fill queue at time {self.conditions.mjd}")

    def get_basis_functions(self, survey_index=None, conditions=None):
        """Get the basis functions for a specific survey,
        in provided conditions.

        Parameters
        ----------
        survey_index : `List` [`int`]
            A list with two elements: the survey list and the element
            within that survey list for which the basis function should be
            retrieved. If ``None``, use the latest survey to make an addition
            to the queue.
        conditions : `rubin_scheduler.scheduler.features.conditions.Conditions`
            The conditions for which to return the basis functions.
            If ``None``, use the conditions associated with this scheduler.

        Returns
        -------
        basis_funcs : `OrderedDict` ['str`,
        `~.scheduler.basis_functions.basis_functions.Base_basis_function`]
            A dictionary of the basis functions, where the keys are names for
            the basis functions and the values are the functions themselves.
        """

        if survey_index is None:
            survey_index = self.survey_index

        if conditions is None:
            conditions = self.conditions

        survey = self.survey_lists[survey_index[0]][survey_index[1]]
        basis_funcs = OrderedDict()
        if hasattr(survey, "basis_functions"):
            for basis_func in survey.basis_functions:
                sample_values = basis_func(conditions)
                if hasattr(sample_values, "__len__"):
                    key = f"{basis_func.__class__.__name__} @{id(basis_func)}"
                    basis_funcs[key] = basis_func
        return basis_funcs

    def get_healpix_maps(self, survey_index=None, conditions=None):
        """Get the healpix maps for a specific survey, in provided conditions.

        Parameters
        ----------
        survey_index : `List` [`int`], opt
            A list with two elements: the survey list and the element within
            that survey list for which the maps that should be retrieved.
            If ``None``, use the latest survey which added to the queue.
        conditions : `~.scheduler.features.conditions.Conditions`, opt
            The conditions for the maps to be returned. If ``None``, use
            the conditions associated with this sceduler. By default None.

        Returns
        -------
        basis_funcs : `OrderedDict` ['str`, `numpy.ndarray`]
            A dictionary of the maps, where the keys are names for the maps
            and values are the numpy arrays as used by ``healpy``.
        """

        if survey_index is None:
            survey_index = self.survey_index

        if conditions is None:
            conditions = self.conditions

        maps = OrderedDict()
        for band in conditions.skybrightness.keys():
            maps[f"{band}_sky"] = deepcopy(conditions.skybrightness[band])
            maps[f"{band}_sky"][maps[f"{band}_sky"] < -1e30] = np.nan

        basis_functions = self.get_basis_functions(survey_index, conditions)

        for this_basis_func in basis_functions.values():
            label = this_basis_func.label()
            if label in maps:
                label = f"{label} @{id(this_basis_func)}"
            maps[label] = this_basis_func(conditions)

        return maps

    def __repr__(self):
        if isinstance(self.pointing2hpindx, HpInLsstFov):
            camera = "LSST"
        elif isinstance(self.pointing2hpindx, HpInComcamFov):
            camera = "comcam"
        else:
            camera = None

        this_repr = f"""{self.__class__.__qualname__}(
            surveys={repr(self.survey_lists)},
            camera="{camera}",
            nside={repr(self.nside)},
            survey_index={repr(self.survey_index)},
            log={repr(self.log)}
        )"""
        return this_repr

    def __str__(self):
        # If dependencies of to_markdown are not installed, fall back on repr
        try:
            pd.DataFrame().to_markdown()
        except ImportError:
            return repr(self)

        if isinstance(self.pointing2hpindx, HpInLsstFov):
            camera = "LSST"
        elif isinstance(self.pointing2hpindx, HpInComcamFov):
            camera = "comcam"
        else:
            camera = None

        output = StringIO()
        print(f"# {self.__class__.__name__} at {hex(id(self))}", file=output)

        try:
            last_chosen = str(self.survey_lists[self.survey_index[0]][self.survey_index[1]])
        except TypeError:
            last_chosen = "None"

        misc = pd.Series(
            {
                "camera": camera,
                "nside": self.nside,
                "survey index": list(self.survey_index),
                "Last chosen": last_chosen,
            }
        )
        misc.name = "value"
        print(misc.to_markdown(), file=output)

        print("", file=output)
        print("## Surveys", file=output)

        if len(self.survey_lists) == 0:
            print("Scheduler contains no surveys.", file=output)

        for tier_index, tier_surveys in enumerate(self.survey_lists):
            print(file=output)
            print(f"### Survey list {tier_index}", file=output)
            print(self.surveys_df(tier_index).to_markdown(), file=output)

        print("", file=output)
        if hasattr(self, "conditions"):
            print(str(self.conditions), file=output)
        else:
            print("No conditions set", file=output)

        print("", file=output)
        print("## Queue", file=output)
        if len(self.queue) > 0:
            print(
                pd.concat(pd.DataFrame(q) for q in self.queue)[
                    ["ID", "flush_by_mjd", "RA", "dec", "filter", "exptime", "note"]
                ]
                .set_index("ID")
                .to_markdown(),
                file=output,
            )
        else:
            print("Queue is empty", file=output)

        result = output.getvalue()
        return result

    def _repr_markdown_(self):
        # This is used by jupyter
        return str(self)

    def surveys_df(self, tier):
        """Create a pandas.DataFrame describing rewards from surveys.

        Parameters
        ----------
        conditions : `rubin_scheduler.scheduler.features.Conditions`
            Conditions for which rewards are to be returned.
        tier : `int`
            The level of the list of survey lists for which to return values.

        Returns
        -------
        reward_df : `pandas.DataFrame`
            A table of surveys listing the rewards.
        """

        surveys = []
        survey_list = self.survey_lists[tier]
        for survey_list_elem, survey in enumerate(survey_list):
            if (self.survey_index[0] is None) or (tier > self.survey_index[0]):
                # This survey reward was not been evaluated
                reward = None
            elif not hasattr(survey, "reward"):
                reward = None
            elif survey.reward is None:
                reward = None
            elif np.isscalar(survey.reward):
                reward = survey.reward
            elif np.count_nonzero(survey.reward > -np.inf) == 0:
                # The entire survey is masked
                reward = -np.inf
            else:
                reward = np.nanmax(survey.reward)

            try:
                chosen = (tier == self.survey_index[0]) and (survey_list_elem == self.survey_index[1])
            except TypeError:
                chosen = False

            surveys.append({"survey": str(survey), "reward": reward, "chosen": chosen})

        df = pd.DataFrame(surveys).set_index("survey")
        return df

    def make_reward_df(self, conditions=None, accum=True):
        """Create a pandas.DataFrame describing rewards from contained surveys.

        Parameters
        ----------
        conditions : `rubin_scheduler.scheduler.features.Conditions`
            Conditions for which rewards are to be returned
        accum : `bool`
            Include accumulated rewards (defaults to True)

        Returns
        -------
        reward_df : `pandas.DataFrame`
            A table of surveys listing the rewards.
        """
        if conditions is None:
            conditions = self.conditions

        survey_dfs = []
        survey_labels = self.survey_labels
        for index0, survey_list in enumerate(self.survey_lists):
            for index1, survey in enumerate(survey_list):
                survey_df = survey.make_reward_df(conditions, accum=accum)
                if len(survey_df) == 0:
                    continue

                survey_df["list_index"] = index0
                survey_df["survey_index"] = index1
                survey_df["tier_label"] = f"tier {index0}"
                survey_df["survey_label"] = survey_labels[index0][index1]
                survey_df["survey_class"] = survey.__class__.__name__
                all_rewards_this_survey = survey.calc_reward_function(conditions)
                survey_df["survey_reward"] = (
                    np.nan
                    if np.all(np.isnan(all_rewards_this_survey))
                    else np.nanmax(all_rewards_this_survey)
                )
                survey_dfs.append(survey_df)

        reward_df = pd.concat(survey_dfs).set_index(["list_index", "survey_index"])
        return reward_df

    @property
    def survey_labels(self):
        """Provide a list of labels for surveys

        Returns
        -------
        label_lists : `list`
            A list, or list of lists, of labels corresponding to
            self.survey_lists
        """
        label_lists = []

        encountered_labels = set()
        duplicated_labels = set()

        def process_survey(survey):
            try:
                basic_label = survey.survey_name
            except AttributeError:
                basic_label = ""

            if len(basic_label) == 0:
                basic_label = survey.__class__.__name__

            if basic_label in encountered_labels:
                duplicated_labels.add(basic_label)
            else:
                encountered_labels.add(basic_label)

            return basic_label

        for item in self.survey_lists:
            if hasattr(item, "generate_observations"):
                basic_label = process_survey(item)
                label_lists.append()
            else:
                labels = []
                for survey in item:
                    basic_label = process_survey(survey)
                    labels.append(basic_label)
                label_lists.append(labels)

        label_count = {}

        def unique_label(survey):
            try:
                basic_label = survey.survey_name
            except AttributeError:
                basic_label = ""

            if len(basic_label) == 0:
                basic_label = survey.__class__.__name__

            if basic_label not in duplicated_labels:
                return basic_label

            if basic_label not in label_count:
                label_count[basic_label] = 0

            label_count[basic_label] += 1
            label = f"{basic_label} {label_count[basic_label]}"
            return label

        label_lists = []
        for item in self.survey_lists:
            if hasattr(item, "generate_observations"):
                label = unique_label(item)
                label_lists.append(label)
            else:
                labels = []
                for survey in item:
                    label = unique_label(survey)
                    labels.append(label)
                label_lists.append(labels)

        return label_lists
