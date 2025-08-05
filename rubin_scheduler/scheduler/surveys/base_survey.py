__all__ = ("BaseSurvey", "BaseMarkovSurvey")

import warnings
from copy import copy, deepcopy
from functools import cached_property

import healpy as hp
import numpy as np
import pandas as pd

from rubin_scheduler.scheduler.detailers import TrackingInfoDetailer, ZeroRotDetailer
from rubin_scheduler.scheduler.utils import (
    ObservationArray,
    comcam_tessellate,
    rotx,
    thetaphi2xyz,
    xyz2thetaphi,
)
from rubin_scheduler.site_models import _read_fields
from rubin_scheduler.utils import (
    DEFAULT_NSIDE,
    _angular_separation,
    _build_tree,
    _hpid2_ra_dec,
    _xyz_from_ra_dec,
)


class BaseSurvey:
    """A baseclass for survey objects.

    Parameters
    ----------
    basis_functions : list
        List of basis_function objects
    extra_features : list XXX--should this be a dict for clarity?
        List of any additional features the survey may want to use
        e.g., for computing final dither positions.
    extra_basis_functions : dict of rubin_scheduler.scheduler.basis_function
        Extra basis function objects. Typically not passed in, but et
        in the __init__.
    ignore_obs : list of str (None)
        If an incoming observation has this string in the note, ignore it.
        Handy if one wants to ignore DD fields or observations
        requested by self. Take note, if a survey is called 'mysurvey23',
        setting ignore_obs to 'mysurvey2' will ignore it because
        'mysurvey2' is a substring of 'mysurvey23'.
    detailers : list of rubin_scheduler.scheduler.detailers objects
        The detailers to apply to the list of observations.
    scheduled_obs : np.array
        An array of MJD values for when observations should execute.
    target_name : `str`
        A target name label. Will be added to a final detailer, so should
        override any target_name set elsewhere. Default None.
    science_program : `str`
        A science program label.  Will be added to a final detailer, so should
        override any science_program set elsewhere. Default None.
    observation_reason : `str`
        An observation reason label.  Will be added to a final detailer, so
        should  override any observation_reason set elsewhere. Default None.
    """

    def __init__(
        self,
        basis_functions,
        extra_features=None,
        extra_basis_functions=None,
        ignore_obs=None,
        survey_name=None,
        scheduler_note=None,
        nside=DEFAULT_NSIDE,
        detailers=None,
        scheduled_obs=None,
        target_name=None,
        science_program=None,
        observation_reason=None,
    ):
        if ignore_obs is None:
            ignore_obs = []

        if isinstance(ignore_obs, str):
            ignore_obs = [ignore_obs]

        self.nside = nside
        if survey_name is None:
            self._generate_survey_name()
        else:
            self.survey_name = survey_name
        self.scheduler_note = scheduler_note
        if self.scheduler_note is None:
            self.scheduler_note = survey_name

        self.ignore_obs = ignore_obs

        self.reward = None
        self.survey_index = None

        self.basis_functions = basis_functions

        if extra_features is None:
            self.extra_features = {}
        else:
            self.extra_features = extra_features
        if extra_basis_functions is None:
            self.extra_basis_functions = {}
        else:
            self.extra_basis_functions = extra_basis_functions

        self.reward_checked = False

        # Attribute to track if the reward function is up-to-date.
        self.reward_checked = False

        # If there's no detailers, add one to set rotation to near zero
        if detailers is None:
            self.detailers = [ZeroRotDetailer(nside=nside)]
        else:
            # If there are detailers -- copy them, because we're
            # about to change the list (and users can be reusing this list).
            self.detailers = deepcopy(detailers)

        # Scheduled observations
        self.scheduled_obs = scheduled_obs

        # Strings to pass onto observations
        if (target_name is not None) | (science_program is not None) | (observation_reason is not None):
            should_have_tracking_detailer = True
        else:
            should_have_tracking_detailer = False
        if should_have_tracking_detailer:
            # Check if one already present - will use that if so.
            has_tracking_detailer = False
            for detailer in self.detailers:
                if isinstance(detailer, TrackingInfoDetailer):
                    has_tracking_detailer = True
                    break
            if has_tracking_detailer:
                warnings.warn(
                    f"Survey {self.survey_name} has a tracking detailer but target_name,"
                    f"observation_reason or science_program also set (and will ignore)."
                )
            else:
                self.detailers.append(
                    TrackingInfoDetailer(
                        target_name=target_name,
                        science_program=science_program,
                        observation_reason=observation_reason,
                    )
                )

    @cached_property
    def roi_hpid(self):
        return None

    @cached_property
    def roi_mask(self):
        if self.roi_hpid is None:
            mask = np.ones(hp.nside2npix(self.nside), dtype=bool)
        else:
            mask = np.zeros(hp.nside2npix(self.nside), dtype=bool)
            mask[self.roi_hpid] = True

        return mask

    def update_conditions(self, conditions):
        """Pass the latest conditions to the survey.

        Usually not relevant, but ScriptedSurveys and ToOs
        may want to update.
        """
        pass

    def _generate_survey_name(self):
        self.survey_name = ""

    def get_scheduled_obs(self):
        return self.scheduled_obs

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

        for feature in self.extra_features:
            self.extra_features[feature].add_observations_array(observations_array, observations_hpid)
        for bf in self.extra_basis_functions:
            self.extra_basis_functions[bf].add_observations_array(observations_array, observations_hpid)
        for bf in self.basis_functions:
            bf.add_observations_array(observations_array, observations_hpid)
        for detailer in self.detailers:
            detailer.add_observations_array(observations_array, observations_hpid)
        self.reward_checked = False

    def add_observation(self, observation, **kwargs):
        # Check each posible ignore string
        checks = [io not in str(observation["scheduler_note"]) for io in self.ignore_obs]
        # ugh, I think here I have to assume observation is an
        # array and not a dict.
        if all(checks):
            for feature in self.extra_features:
                self.extra_features[feature].add_observation(observation, **kwargs)
            for bf in self.extra_basis_functions:
                self.extra_basis_functions[bf].add_observation(observation, **kwargs)
            for bf in self.basis_functions:
                bf.add_observation(observation, **kwargs)
            for detailer in self.detailers:
                detailer.add_observation(observation, **kwargs)
            self.reward_checked = False

    def _check_feasibility(self, conditions):
        """
        Check if the survey is feasible in the current conditions

        Returns
        -------
        result : `bool`
        """
        result = True
        for bf in self.basis_functions:
            result = bf.check_feasibility(conditions)
            if not result:
                return result
        return result

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
        else:
            # If we don't pass feasability
            self.reward = -np.inf

        self.reward_checked = True
        return self.reward

    def generate_observations_rough(self, conditions):
        """
        Returns
        -------
        ObservationArray
        """
        # If the reward function hasn't been updated with the
        # latest info, calculate it
        if not self.reward_checked:
            self.reward = self.calc_reward_function(conditions)
        obs = ObservationArray()
        return obs

    def generate_observations(self, conditions):
        observations = self.generate_observations_rough(conditions)
        if np.size(observations) > 0:
            for detailer in self.detailers:
                observations = detailer(observations, conditions)
        return observations

    def viz_config(self):
        # This primarily lives in schedview but could be somewhat added here
        pass

    def __repr__(self):
        try:
            repr = f"<{self.__class__.__name__} survey_name='{self.survey_name}' at {hex(id(self))}>"
        except AttributeError:
            repr = f"<{self.__class__.__name__} at {hex(id(self))}>"

        return repr

    def _reward_to_scalars(self, reward):
        if np.isscalar(reward):
            scalar_reward = reward
            if np.isnan(reward) or reward == -np.inf:
                unmasked_area = 0
                scalar_reward = -np.inf
            else:
                try:
                    pix_area = hp.nside2pixarea(self.nside, degrees=True)
                    unmasked_area = pix_area * np.count_nonzero(self.roi_mask)
                except AttributeError:
                    unmasked_area = 4 * np.pi * (180 / np.pi) ** 2
        else:
            reward_in_roi = np.where(self.roi_mask, reward, -np.inf)
            pix_area = hp.nside2pixarea(self.nside, degrees=True)
            unmasked_area = pix_area * np.count_nonzero(reward_in_roi > -np.inf)
            if unmasked_area == 0:
                scalar_reward = -np.inf
            else:
                scalar_reward = np.nanmax(reward_in_roi)

        return scalar_reward, unmasked_area

    def make_reward_df(self, conditions, accum=True):
        """Create a pandas.DataFrame describing the reward from the survey.

        Parameters
        ----------
        conditions : `rubin_scheduler.scheduler.features.Conditions`
            Conditions for which rewards are to be returned
        accum : `bool`
            Include accumulated reward (more compute intensive)
            Defaults to True

        Returns
        -------
        reward_df : `pandas.DataFrame`
            A table of surveys listing the rewards.
        """

        feasibility = []
        max_rewards = []
        basis_areas = []
        accum_rewards = []
        accum_areas = []
        bf_label = []
        bf_class = []
        basis_functions = []
        basis_weights = []

        try:
            full_basis_weights = self.basis_weights
        except AttributeError:
            full_basis_weights = [1.0 for df in self.basis_functions]

        short_labels = self.bf_short_labels

        _, scalar_area = self._reward_to_scalars(1)

        for weight, basis_function in zip(full_basis_weights, self.basis_functions):
            bf_label.append(short_labels[basis_function.label()])
            bf_class.append(basis_function.__class__.__name__)
            bf_reward = basis_function(conditions)
            max_reward, basis_area = self._reward_to_scalars(bf_reward)

            if basis_area == 0:
                this_feasibility = False
            else:
                this_feasibility = np.array(basis_function.check_feasibility(conditions)).any()

            feasibility.append(this_feasibility)
            max_rewards.append(max_reward)
            basis_areas.append(basis_area)

            if accum:
                basis_functions.append(basis_function)
                basis_weights.append(weight)
                test_survey = deepcopy(self)
                test_survey.basis_functions = basis_functions
                test_survey.basis_weights = basis_weights
                this_accum_reward = test_survey.calc_reward_function(conditions)
                accum_reward, accum_area = self._reward_to_scalars(this_accum_reward)
                accum_rewards.append(accum_reward)
                accum_areas.append(accum_area)

        reward_data = {
            "basis_function": bf_label,
            "basis_function_class": bf_class,
            "feasible": feasibility,
            "max_basis_reward": max_rewards,
            "basis_area": basis_areas,
            "basis_weight": full_basis_weights,
        }
        if accum:
            reward_data["max_accum_reward"] = accum_rewards
            reward_data["accum_area"] = accum_areas

        reward_df = pd.DataFrame(reward_data)

        return reward_df

    def reward_changes(self, conditions):
        """List the rewards for each basis function used by the survey.

        Parameters
        ----------
        conditions : `rubin_scheduler.scheduler.features.Conditions`
            Conditions for which rewards are to be returned

        Returns
        -------
        rewards : `list`
            A list of tuples, each with a basis function name and the
            maximum reward returned by that basis function for the
            provided conditions.
        """

        reward_values = []
        basis_functions = []
        basis_weights = []

        try:
            full_basis_weights = self.basis_weights
        except AttributeError:
            full_basis_weights = [1 for bf in self.basis_functions]

        for weight, basis_function in zip(full_basis_weights, self.basis_functions):
            test_survey = deepcopy(self)
            basis_functions.append(basis_function)
            test_survey.basis_functions = basis_functions
            basis_weights.append(weight)
            test_survey.basis_weights = basis_weights
            try:
                reward_values.append(np.nanmax(test_survey.calc_reward_function(conditions)))
            except IndexError:
                reward_values.append(None)

        bf_names = [bf.label() for bf in self.basis_functions]
        return list(zip(bf_names, reward_values))

    @property
    def bf_short_labels(self):
        try:
            long_labels = [bf.label() for bf in self.basis_functions]
        except AttributeError:
            return []

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


class BaseMarkovSurvey(BaseSurvey):
    """A Markov Decision Function survey object. Uses Basis functions
    to compute a final reward function and decide what to observe based
    on the reward. Includes methods for dithering and defaults to
    dithering nightly.

    Parameters
    ----------
    basis_function : list of rubin_scheduler.scheduler.basis_function

    basis_weights : list of float
        Must be same length as basis_function
    seed : hashable
        Random number seed, used for randomly orienting sky tessellation.
    camera : str ('LSST')
        Should be 'LSST' or 'comcam'
    fields : np.array (None)
        An array of field positions. Should be numpy array with columns
        of "RA" and "dec" in radians. If none,
        site_models.read_fields or utils.comcam_tessellate is used to
        read field positions.
    area_required : float (None)
        The valid area that should be present in the reward
        function (square degrees).
    npositions : int (7305)
        The number of dither positions to pre-compute. Defaults
        to 7305 (so good for 20 years)
    dither : `str`
        Possible values of "night" (default), "call", or None.
        Spins sky tesselation per night, per call, or not at all.
    """

    def __init__(
        self,
        basis_functions,
        basis_weights,
        extra_features=None,
        smoothing_kernel=None,
        ignore_obs=None,
        survey_name=None,
        scheduler_note=None,
        nside=DEFAULT_NSIDE,
        seed=42,
        dither="night",
        detailers=None,
        camera="LSST",
        fields=None,
        area_required=None,
        npositions=7305,
        target_name=None,
        science_program=None,
        observation_reason="Area",
    ):
        super(BaseMarkovSurvey, self).__init__(
            basis_functions=basis_functions,
            extra_features=extra_features,
            ignore_obs=ignore_obs,
            survey_name=survey_name,
            scheduler_note=scheduler_note,
            nside=nside,
            detailers=detailers,
            target_name=target_name,
            science_program=science_program,
            observation_reason=observation_reason,
        )

        self.basis_weights = basis_weights
        self.current_night = 0
        # Check that weights and basis functions are same length
        if len(basis_functions) != np.size(basis_weights):
            raise ValueError("basis_functions and basis_weights must be same length.")

        self.camera = camera
        # Load the OpSim field tesselation and map healpix to fields
        if fields is None:
            if self.camera == "LSST":
                ra, dec = _read_fields()
                self.fields_init = np.empty(ra.size, dtype=list(zip(["RA", "dec"], [float, float])))
                self.fields_init["RA"] = ra
                self.fields_init["dec"] = dec
            elif self.camera == "comcam":
                self.fields_init = comcam_tessellate()
            else:
                ValueError('camera %s unknown, should be "LSST" or "comcam"' % camera)
        else:
            self.fields_init = fields
        self.fields = self.fields_init.copy()
        self._hp2fieldsetup(self.fields["RA"], self.fields["dec"])

        if smoothing_kernel is not None:
            self.smoothing_kernel = np.radians(smoothing_kernel)
        else:
            self.smoothing_kernel = None

        if area_required is None:
            self.area_required = area_required
        else:
            self.area_required = area_required * (np.pi / 180.0) ** 2  # To steradians

        # Start tracking the night
        self.night = -1

        if dither in [True, False]:
            survey_name = self.survey_name
            if len(survey_name) == 0:
                survey_name = self.__class__.__name__
            warnings.warn(
                f"setting dither to bool deprecated, swapping to dither='night' ({survey_name})",
                FutureWarning,
            )
            dither = "night"
        if dither not in ["night", "call", None]:
            raise ValueError("dither kwarg must be one of 'night', 'call', or None.")
        self.dither = dither
        self.call_num = 0

        # Generate and store rotation positions to use.
        # This way, if different survey objects are seeded the same,
        # they will use the same dither positions each night
        if dither == "call":
            npositions = (npositions, npositions)
        rng = np.random.default_rng(seed)
        self.lon = rng.random(npositions) * np.pi * 2
        # Make sure latitude points spread correctly
        # http://mathworld.wolfram.com/SpherePointPicking.html
        self.lat = np.arccos(2.0 * rng.random(npositions) - 1.0)
        self.lon2 = rng.random(npositions) * np.pi * 2

    def _check_feasibility(self, conditions):
        """
        Check if the survey is feasable in the current conditions
        """
        for bf in self.basis_functions:
            result = bf.check_feasibility(conditions)
            if not result:
                return result
        # If we need to check that the reward function has enough
        # area available
        if self.area_required is not None:
            reward = self.calc_reward_basic(conditions)
            # Are there any valid reward pixels remaining
            if np.sum(np.isfinite(reward)) > 0:
                max_reward_indx = np.min(np.where(reward == np.nanmax(reward)))
                distances = _angular_separation(
                    self.ra, self.dec, self.ra[max_reward_indx], self.dec[max_reward_indx]
                )
                valid_pix = np.where(np.logical_not(np.isnan(reward)) & (distances < self.max_radius_peak))[0]
                if np.size(valid_pix) * self.pixarea < self.area_required:
                    result = False
            else:
                result = False
        return result

    def _hp2fieldsetup(self, ra, dec):
        """Map each healpixel to nearest field. This will only work
        if healpix resolution is higher than field resolution.

        Parameters
        ----------
        ra : `float`
            The RA of the possible pointings (radians)
        dec : `float`
            The decs of the possible pointings (radians)
        """

        # Let's just map each healpix to the closest field location
        tree = _build_tree(ra, dec)
        hp_ra, hp_dec = _hpid2_ra_dec(self.nside, np.arange(hp.nside2npix(self.nside)))
        x, y, z = _xyz_from_ra_dec(hp_ra, hp_dec)
        dist, ind = tree.query(np.vstack([x, y, z]).T, k=1)

        # XXX--maybe add a check that distance isn't too large
        self.hp2fields = ind

    def _spin_fields(self, indx, lon=None, lat=None, lon2=None):
        """Spin the field tessellation to generate a random orientation

        The default field tesselation is rotated randomly in longitude,
        and then the pole is rotated to a random point on the sphere.

        Parameters
        ----------
        lon : float (None)
            The amount to initially rotate in longitude (radians).
            Will use a random value
            between 0 and 2 pi if None (default).
        lat : float (None)
            The amount to rotate in latitude (radians).
        lon2 : float (None)
            The amount to rotate the pole in longitude (radians).
        """
        if lon is None:
            lon = self.lon[indx]
        if lat is None:
            lat = self.lat[indx]
        if lon2 is None:
            lon2 = self.lon2[indx]

        # rotate longitude
        ra = (self.fields_init["RA"] + lon) % (2.0 * np.pi)
        dec = copy(self.fields_init["dec"])

        # Now to rotate ra and dec about the x-axis
        x, y, z = thetaphi2xyz(ra, dec + np.pi / 2.0)
        xp, yp, zp = rotx(lat, x, y, z)
        theta, phi = xyz2thetaphi(xp, yp, zp)
        dec = phi - np.pi / 2
        ra = theta + np.pi

        # One more RA rotation
        ra = (ra + lon2) % (2.0 * np.pi)

        self.fields["RA"] = ra
        self.fields["dec"] = dec
        # Rebuild the kdtree with the new positions
        # XXX-may be doing some ra,dec to conversions xyz more
        # than needed.
        self._hp2fieldsetup(ra, dec)

    def smooth_reward(self):
        """If we want to smooth the reward function."""
        if hp.isnpixok(self.reward.size):
            # Need to swap NaNs to hp.UNSEEN so smoothing doesn't
            # spread mask
            reward_temp = copy(self.reward)
            mask = np.isnan(reward_temp)
            reward_temp[mask] = hp.UNSEEN
            self.reward_smooth = hp.sphtfunc.smoothing(reward_temp, fwhm=self.smoothing_kernel, verbose=False)
            self.reward_smooth[mask] = np.nan
            self.reward = self.reward_smooth

    def calc_reward_basic(self, conditions):
        reward = 0
        indx = np.arange(hp.nside2npix(self.nside))
        for bf, weight in zip(self.basis_functions, self.basis_weights):
            basis_value = bf(conditions, indx=indx)
            reward += basis_value * weight
        return reward

    def calc_reward_function(self, conditions):
        self.reward_checked = True
        if self._check_feasibility(conditions):
            self.reward = self.calc_reward_basic(conditions)
        else:
            # If not feasable, negative infinity reward
            self.reward = -np.inf
            return self.reward
        if self.smoothing_kernel is not None:
            self.smooth_reward()

        if self.area_required is not None:
            good_area = np.where(np.abs(self.reward) >= 0)[0].size * hp.nside2pixarea(self.nside)
            if good_area < self.area_required:
                self.reward = -np.inf

        return self.reward

    def generate_observations_rough(self, conditions):
        self.reward = self.calc_reward_function(conditions)

        # Check if we need to spin the tesselation
        if (self.dither == "night") & (conditions.night != self.night):
            self._spin_fields(conditions.night)
            self.night = copy(conditions.night)
        elif self.dither == "call":
            if conditions.night != self.current_night:
                self.current_night = conditions.night
                self.call_num = 0
            self._spin_fields((conditions.night, self.call_num))
            self.call_num += 1

        # XXX Use self.reward to decide what to observe.
        return None
