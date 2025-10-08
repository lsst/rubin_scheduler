__all__ = (
    "BaseDetailer",
    "ZeroRotDetailer",
    "Comcam90rotDetailer",
    "Rottep2RotspDesiredDetailer",
    "CloseAltDetailer",
    "TakeAsPairsDetailer",
    "TwilightTripleDetailer",
    "FlushForSchedDetailer",
    "FilterNexp",
    "BandNexp",
    "FixedSkyAngleDetailer",
    "ParallacticRotationDetailer",
    "FlushByDetailer",
    "RandomFilterDetailer",
    "RandomBandDetailer",
    "BandSortDetailer",
    "TruncatePreTwiDetailer",
    "TrackingInfoDetailer",
    "AltAz2RaDecDetailer",
    "StartFieldSequenceDetailer",
    "BandToFilterDetailer",
    "TagRadialDetailer",
    "CopyValueDetailer",
    "LabelRegionDetailer",
    "LabelDDFDetailer",
    "LabelRegionsAndDDFs",
    "RollBandMatchDetailer",
)

import copy
import warnings

import healpy as hp
import numpy as np

import rubin_scheduler.scheduler.features as features
from rubin_scheduler.scheduler.utils import CurrentAreaMap, HpInLsstFov, IntRounded, ObservationArray
from rubin_scheduler.utils import (
    DEFAULT_NSIDE,
    _angular_separation,
    _approx_alt_az2_ra_dec,
    _approx_altaz2pa,
    _approx_ra_dec2_alt_az,
    ddf_locations,
    pseudo_parallactic_angle,
    rotation_converter,
)


class BaseDetailer:
    """
    A Detailer is an object that takes a list of proposed observations and
    adds "details" to them. The primary purpose is that the Markov Decision
    Process does an excelent job selecting RA,Dec,band combinations, but we
    may want to add additional logic such as what to set the camera rotation
    angle to, or what to use for an exposure time. We could also modify the
    order of the proposed observations. For Deep Drilling Fields, a detailer
    could be useful for computing dither positions and modifying the exact
    RA,Dec positions.
    """

    def __init__(self, nside=DEFAULT_NSIDE):
        """"""
        # Dict to hold all the features we want to track
        self.survey_features = {}
        self.nside = nside

    def add_observations_array(self, observations_array, observations_hpid):
        """Like add_observation, but for loading a whole array of
        observations at a time"""
        if hasattr(self, "survey_features"):
            for feature in self.survey_features:
                self.survey_features[feature].add_observations_array(observations_array, observations_hpid)

    def add_observation(self, observation, indx=None):
        """
        Parameters
        ----------
        observation : `np.array`
            An array with information about the input observation
        indx : `np.array`
            The indices of the healpix map that the observation overlaps
            with
        """
        if hasattr(self, "survey_features"):
            for feature in self.survey_features:
                self.survey_features[feature].add_observation(observation, indx=indx)

    def __call__(self, observation_list, conditions):
        """
        Parameters
        ----------
        observation_list : `list` of observations
            The observations to detail.
        conditions : `rubin_scheduler.scheduler.conditions` object

        Returns
        -------
        ObservationArray
        """

        return ObservationArray()


class TrackingInfoDetailer(BaseDetailer):
    """Add metadata related to target_name, observation_reason and
    target_name for an observation.

    Does NOT clobber information that has been previously set.
    Appended to all surveys detailer lists, if not already present,
    by BaseSurvey using values from survey configuration.

    See https://rtn-096.lsst.io for further information on these observation
    metadata fields.

    Parameters
    ----------
    target_name : `str` or `None`
        The value to enter into the target_name field of the Observation.
        With current metadata expectations, this will indicate a specific
        target name or a general region label. See also LabelRegionDetailer.
    science_program : `str` or `None`
        The value to enter into the science_program field of the Observation.
        With the current configuration of the SchedulerCSC, this corresponds
        to the JSON BLOCK executed to acquire the visit.
    observation_reason : `str` or `None`
        The value to enter into the observation_reason of the Observation.
        With current metadata expectations, this will indicate something
        about the observing mode.
        Spaces in observation_reason will be replaced by underscores,
        required by camera handling of metadata.
    """

    def __init__(self, target_name="", science_program="", observation_reason=""):
        self.survey_features = {}
        self.target_name = target_name
        self.science_program = science_program
        self.observation_reason = observation_reason
        if self.observation_reason is not None:
            self.observation_reason = self.observation_reason.replace(" ", "_")
        self.keys = ["science_program", "target_name", "observation_reason"]

    def __call__(self, observation_array, conditions):
        for key in self.keys:
            indx = np.where(
                (observation_array[key] == "")
                | (observation_array[key] is None)
                | (observation_array[key] == "None")
            )[0]
            observation_array[key][indx] = getattr(self, key)

        return observation_array


class RollBandMatchDetailer(BaseDetailer):
    """Roll the order of visits to eliminate a filter change."""

    def __call__(self, observation_array, conditions):

        bm_index = np.where(observation_array["band"] == conditions.current_band)[0]
        if np.size(bm_index) > 0:
            indx = np.arange(observation_array.size)
            indx = np.roll(indx, -np.min(bm_index))
            observation_array = observation_array[indx]

        return observation_array


class FlushByDetailer(BaseDetailer):
    """Set the MJD an observation should be flushed from the scheduler
    queue if not yet completed.

    Parameters
    ----------
    flush_time : `float`
        The time to flush after the current MJD. Default 60 minutes
    """

    def __init__(self, flush_time=60):
        self.survey_features = {}
        self.flush_time = flush_time / 60 / 24.0

    def __call__(self, observation_array, conditions):
        observation_array["flush_by_mjd"] = conditions.mjd + self.flush_time
        return observation_array


class BandToFilterDetailer(BaseDetailer):
    """If we want to fill in the physical filter to request rather
    than just the band.

    Parameters
    ----------
    band_to_filter_dict : `dict`
        A dict that maps band name (usually ugrizy) to
        specific filter names. Default value of None
        will set the filter name to the same as the band name.
    """

    def __init__(self, band_to_filter_dict=None):
        self.survey_features = {}
        if band_to_filter_dict is None:
            self.band_to_filter_dict = {}
            for bandname in "ugrizy":
                self.band_to_filter_dict[bandname] = bandname
        else:
            self.band_to_filter_dict = band_to_filter_dict

    def __call__(self, observation_array, conditions):
        u_bands = np.unique(observation_array["band"])
        for band in u_bands:
            indx = np.where(observation_array["band"] == band)
            # Fetch the dictionary value or just continue to use band
            filtername = self.band_to_filter_dict.get(band, band)
            observation_array["filter"][indx] = filtername

        return observation_array


class TruncatePreTwiDetailer(BaseDetailer):
    """Truncate an array of observations so they fit before
    morning twilight starts.

    Parameters
    ----------
    pad : `float`
        The pad to give before start of morning twilight.
        Default 5 (minutes).
    filter_change_time : `float`
        Estimate of filter change time
    visit_overhead : `float`
        Estimate of visit overhead (read, slew, settle).
        Default 4.0 (seconds).
    twilight_cut_to : `str`
        Cut to sun altitude of -18 degrees or -12 degrees.
        Valid values of "n18" or "n12".
    """

    def __init__(self, pad=5.0, filter_change_time=120.0, visit_overhead=4.0, twilight_cut_to="n18"):
        self.pad = pad / 60.0 / 24.0  # To days
        self.filter_change_time = filter_change_time / 3600.0 / 24.0  # To days

        self.visit_overhead = visit_overhead / 3600.0 / 24.0  # To days
        self.twilight_cut_to = twilight_cut_to

    def __call__(self, observation_array, conditions):

        b1 = observation_array["band"][0:-1]
        b2 = observation_array["band"][1:]

        f_changes = b1 != b2
        # Do we start with a filter change?
        prepend = observation_array["band"][0] != conditions.current_band
        f_changes = np.array([prepend] + np.array(f_changes).tolist())
        f_time = f_changes * self.filter_change_time

        tot_time = observation_array["exptime"] / 3600.0 / 24.0 + f_time + self.visit_overhead
        cumulative_time = np.cumsum(tot_time)

        cut_mjd = getattr(conditions, "sun_%s_rising" % self.twilight_cut_to)

        time_avail = cut_mjd - conditions.mjd - self.pad

        indx = np.where(IntRounded(cumulative_time) <= IntRounded(time_avail))[0]
        if np.size(indx) > 0:
            trunc_indx = np.max(indx)
            return observation_array[0 : trunc_indx + 1]
        else:
            return ObservationArray(n=0)


class AltAz2RaDecDetailer(BaseDetailer):
    """Set RA,dec for an observation that only has alt,az"""

    def __call__(self, observation_array, conditions):
        ra, dec = _approx_alt_az2_ra_dec(
            observation_array["alt"],
            observation_array["az"],
            conditions.site.latitude_rad,
            conditions.site.longitude_rad,
            conditions.mjd,
        )
        observation_array["RA"] = ra
        observation_array["dec"] = dec

        return observation_array


class RandomFilterDetailer(BaseDetailer):
    """Deprecated in favor of RandomBandDetailer"""

    def __init__(self, filters="riz", nights_to_prep=10000, seed=42, fallback_order="rizgyu"):
        warnings.warn("Deprecated in favor of RandomBandDetailer", FutureWarning)
        super().__init__(
            bands=filters, nights_to_prep=nights_to_prep, seed=seed, fallback_order=fallback_order
        )


class RandomBandDetailer(BaseDetailer):
    """Pick a random band for the observations

    Parameters
    ----------
    bands : `str`
        The bands to randomize. Default 'riz'
    nights_to_prep : `int`
        The number of nights to generate random bands for.
        Default 10000.
    seed : number
        Seed for RNG. Defaut 42
    fallback_order : `str`
        If the desired band is not mounted, goes through
        `fallback_order` and uses the first band that is
        available
    """

    def __init__(self, bands="riz", nights_to_prep=10000, seed=42, fallback_order="rizgyu"):
        self.survey_features = []
        self.fallback_order = fallback_order

        rng = np.random.default_rng(seed)
        self.night2band_int = rng.integers(low=0, high=len(bands), size=nights_to_prep)

        self.band_dict = {}
        for i, bandname in enumerate(bands):
            self.band_dict[i] = bandname

    def __call__(self, observation_array, conditions):
        band_to_use = self.band_dict[self.night2band_int[conditions.night]]
        # Band not available
        if band_to_use not in conditions.mounted_bands:
            is_mounted = [bandname in conditions.mounted_bands for bandname in self.fallback_order]
            indx = np.min(np.where(is_mounted))
            band_to_use = self.fallback_order[indx]

        observation_array["band"] = band_to_use
        return observation_array


class BandSortDetailer(BaseDetailer):
    """Detailer that sorts an array of observations by specified
    band order, in order to minimize filter changes.

    Useful for scripted surveys with many filter changes like DDFs.

    Parameters
    ----------
    desired_band_order : `str`
        The desired band order. Default of None retains order
        and only moves observations up if they match the current filter.
    loaded_first : `bool`
        If True, then the currently-in-use filter is always moved to
        the start of the desired band order, to remove the first filter change.
    """

    def __init__(
        self,
        desired_band_order: str = None,
        loaded_first: bool = True,
    ):
        self.desired_band_order = desired_band_order
        self.loaded_first = loaded_first

    def __call__(
        self, observation_array: ObservationArray, conditions: features.Conditions
    ) -> ObservationArray:

        order_to_set = copy.copy(self.desired_band_order)

        u_vals, indx = np.unique(observation_array["band"], return_index=True)
        # If we have only 1 filter, nothing to sort, so return
        if np.size(u_vals) == 1:
            return observation_array

        if order_to_set is None:
            # unique values, in order they appear, as a string
            order_to_set = "".join(u_vals[np.argsort(indx)].tolist())

        if self.loaded_first:
            order_to_set = order_to_set.replace(conditions.current_band, "")
            order_to_set = conditions.current_band + order_to_set
        indicies = []
        for bandname in order_to_set:
            indicies.append(np.where(observation_array["band"] == bandname)[0])

        indices = np.concatenate(indicies)

        return observation_array[indices]


class ParallacticRotationDetailer(BaseDetailer):
    """Set the rotator to near the parallactic angle"""

    def __init__(self, telescope="rubin"):
        self.rc = rotation_converter(telescope=telescope)
        self.survey_features = {}

    def __call__(self, observation_array, conditions, limits=[-270, 270]):
        limits = np.radians(limits)

        alt, az = _approx_ra_dec2_alt_az(
            observation_array["RA"],
            observation_array["dec"],
            conditions.site.latitude_rad,
            conditions.site.longitude_rad,
            conditions.mjd,
        )
        obs_pa = _approx_altaz2pa(alt, az, conditions.site.latitude_rad)
        observation_array["rotSkyPos_desired"] = obs_pa

        resulting_rot_tel_pos = self.rc._rotskypos2rottelpos(observation_array["rotSkyPos_desired"], obs_pa)

        indx = np.where(resulting_rot_tel_pos > np.max(limits))[0]
        resulting_rot_tel_pos[indx] -= 2 * np.pi

        indx = np.where(resulting_rot_tel_pos < np.min(limits))
        resulting_rot_tel_pos[indx] += 2 * np.pi

        # If those corrections still leave us bad, just pull it back 180.
        indx = np.where(resulting_rot_tel_pos > np.max(limits))
        resulting_rot_tel_pos[indx] -= np.pi

        # The rotTelPos overides everything else.
        observation_array["rotTelPos"] = resulting_rot_tel_pos
        # if the rotSkyPos_desired isn't possible, fall back to this.
        observation_array["rotTelPos_backup"] = 0

        return observation_array


class Rottep2RotspDesiredDetailer(BaseDetailer):
    """Convert all the rotTelPos values to rotSkyPos_desired"""

    def __init__(self, telescope="rubin"):
        self.rc = rotation_converter(telescope=telescope)
        self.survey_features = {}

    def __call__(self, obs_array, conditions):
        alt, az = _approx_ra_dec2_alt_az(
            obs_array["RA"],
            obs_array["dec"],
            conditions.site.latitude_rad,
            conditions.site.longitude_rad,
            conditions.mjd,
        )
        obs_pa = _approx_altaz2pa(alt, az, conditions.site.latitude_rad)

        rot_sky_pos_desired = self.rc._rotskypos2rottelpos(obs_array["rotTelPos"], obs_pa)

        obs_array["rotTelPos_backup"] = obs_array["rotTelPos"] + 0
        obs_array["rotTelPos"] = np.nan
        obs_array["rotSkyPos"] = np.nan
        obs_array["rotSkyPos_desired"] = rot_sky_pos_desired

        return obs_array


class ZeroRotDetailer(BaseDetailer):
    """
    Detailer to set the camera rotation to be apporximately zero in
    rotTelPos.

    Parameters
    ----------
    telescope : `str`
        Which telescope convention to use for setting the conversion
        between rotTelPos and rotSkyPos. Default "rubin".
    """

    def __init__(self, telescope="rubin", nside=DEFAULT_NSIDE):
        self.rc = rotation_converter(telescope=telescope)
        self.survey_features = {}

    def __call__(self, observation_array, conditions):
        obs_pa, alt, az = pseudo_parallactic_angle(
            observation_array["RA"],
            observation_array["dec"],
            conditions.mjd,
            np.degrees(conditions.site.longitude_rad),
            np.degrees(conditions.site.latitude_rad),
        )

        observation_array["rotSkyPos"] = np.radians(self.rc.rottelpos2rotskypos(0.0, obs_pa))

        return observation_array


class Comcam90rotDetailer(BaseDetailer):
    """
    Detailer to set the camera rotation so rotSkyPos is 0, 90, 180, or
    270 degrees. Whatever is closest to rotTelPos of zero.
    """

    def __init__(self, telescope="rubin", nside=DEFAULT_NSIDE):
        self.rc = rotation_converter(telescope=telescope)
        self.survey_features = {}

    def __call__(self, obs_array, conditions):
        favored_rot_sky_pos = np.radians([0.0, 90.0, 180.0, 270.0, 360.0]).reshape(5, 1)

        parallactic_angle, alt, az = np.radians(
            pseudo_parallactic_angle(
                obs_array["RA"],
                obs_array["dec"],
                conditions.mjd,
                np.degrees(conditions.site.longitude_rad),
                np.degrees(conditions.site.latitude_rad),
            )
        )

        # need to find the

        ang_diff = np.abs(self.rc._rotskypos2rottelpos(favored_rot_sky_pos, np.radians(parallactic_angle)))
        min_indxs = np.argmin(ang_diff, axis=0)
        # can swap 360 and zero if needed?
        obs_array["rotSkyPos"] = favored_rot_sky_pos[min_indxs].ravel()

        return obs_array


class StartFieldSequenceDetailer(BaseDetailer):
    """Prepend a sequence of observations to the start of an array

    Parameters
    ----------
    sequence_obs : `ObservationArray`
        ObservationArray object. The observations should
        have "scheduler_note" and/or "science_program" set.
    ang_distance_match : `float`
        How close should an observation be on the sky to be considered
        matching (degrees).
    time_match_hours : `float`
        How close in time to demand an observation be matching (hours).
    science_program : `str` or None
        The science_program to match against. Default None.
    scheduler_note : `str` or None
        The scheduler_note to match observations against.
        Default "starting_sequence".
    ra : `float`
        RA to match against. Default 0 (degrees). Ignored
        if ang_distance_match is None.
    dec : `float`
        Dec to match observations against. Default 0 (degrees).
        Ignored if ang_distance_match is None.

    """

    def __init__(
        self,
        sequence_obs,
        ang_distance_match=3.5,
        time_match_hours=5,
        science_program=None,
        scheduler_note="starting_sequence",
        ra=0,
        dec=0,
    ):
        super().__init__()
        self.survey_features["last_matching"] = features.LastObservedMatching(
            ang_distance_match=3.5,
            science_program=science_program,
            scheduler_note=scheduler_note,
            ra=ra,
            dec=dec,
        )
        self.ang_distance_match = np.radians(ang_distance_match)
        self.time_match = time_match_hours / 24.0
        self.science_program = science_program
        self.scheduler_note = scheduler_note

        # Make backwards compatible if someone sent in a list
        if isinstance(sequence_obs, list):
            warnings.warn("sequence_obs should be ObsArray, not list of ObsArray. Concatenating")
            sequence_obs = np.concatenate(sequence_obs)

        self.sequence_obs = sequence_obs

        self.sequence_obs["science_program"] = self.science_program
        self.sequence_obs["scheduler_note"] = self.scheduler_note

        # Check that things are sensibly set

        u_scip = np.unique(sequence_obs["science_program"])
        u_sn = np.unique(sequence_obs["scheduler_note"])

        if (np.size(u_scip) > 1) | (np.size(u_sn) > 1):
            msg = (
                "The science_program and/or scheduler_note " "values in sequence_obs_list should be the same."
            )
            raise ValueError(msg)

        if science_program is not None:
            if self.science_program != u_scip:
                ValueError("science_program kwarg not equal to science_programs from sequence_obs_list")
        if scheduler_note is not None:
            if self.scheduler_note != u_sn:
                ValueError("scheduler_note kwarg not equal to scheduler_notes from sequence_obs_list")

    def __call__(self, observation_array, conditions):
        # Do we need to add the opening sequence?
        if (conditions.mjd - self.survey_features["last_matching"].feature["mjd"]) >= self.time_match:
            observation_array = np.concatenate([self.sequence_obs, observation_array])

        return observation_array


class FixedSkyAngleDetailer(BaseDetailer):
    """Detailer to force a specific sky angle.

    Parameters
    ----------
    sky_angle : `float`, optional
        Desired sky angle (default = 0, in degrees).
    """

    def __init__(self, sky_angle=0.0, nside=DEFAULT_NSIDE):
        super().__init__(nside=nside)

        self.sky_angle = np.radians(sky_angle)

    def __call__(self, observation_array, conditions):
        observation_array["rotSkyPos"] = self.sky_angle

        return observation_array


class CloseAltDetailer(BaseDetailer):
    """Re-order a list of observations so that the closest in altitude to
    the current pointing is first.

    Parameters
    ----------
    alt_band : `float` (10)
        The altitude band to try and stay in (degrees)
    """

    def __init__(self, alt_band=10.0):
        super(CloseAltDetailer, self).__init__()
        self.alt_band = IntRounded(np.radians(alt_band))

    def __call__(self, obs_array, conditions):
        alt, az = _approx_ra_dec2_alt_az(
            obs_array["RA"],
            obs_array["dec"],
            conditions.site.latitude_rad,
            conditions.site.longitude_rad,
            conditions.mjd,
        )
        alt_diff = np.abs(alt - conditions.tel_alt)
        in_band = np.where(IntRounded(alt_diff) <= self.alt_band)[0]
        if in_band.size == 0:
            in_band = np.arange(alt.size)

        # Find the closest in angular distance of the points that are in band
        ang_dist = _angular_separation(az[in_band], alt[in_band], conditions.tel_az, conditions.tel_alt)
        if np.size(ang_dist) == 1:
            good = 0
        else:
            good = np.min(np.where(ang_dist == ang_dist.min())[0])
        indx = in_band[good]
        result = np.concatenate([obs_array[indx:], obs_array[:indx]])
        return result


class FlushForSchedDetailer(BaseDetailer):
    """Update the flush-by MJD to be before any scheduled observations

    Parameters
    ----------
    tol : `float`
         How much before to flush (minutes)
    """

    def __init__(self, tol=2.5):
        super(FlushForSchedDetailer, self).__init__()
        self.tol = tol / 24.0 / 60.0  # To days

    def __call__(self, observation_array, conditions):
        if np.size(conditions.scheduled_observations) > 0:
            new_flush = np.min(conditions.scheduled_observations) - self.tol
            indx = np.where(observation_array["flush_by_mjd"] > new_flush)[0]
            observation_array[indx]["flush_by_mjd"] = new_flush
        return observation_array


class BandNexp(BaseDetailer):
    """Demand one band always be taken as a certain number of exposures"""

    def __init__(self, bandname="u", nexp=1, exptime=None):
        super(BandNexp, self).__init__()
        self.bandname = bandname
        self.nexp = nexp
        self.exptime = exptime

    def __call__(self, observation_array, conditions):
        indx = np.where(observation_array["band"] == self.bandname)[0]
        observation_array["nexp"][indx] = self.nexp
        if self.exptime is not None:
            observation_array["exptime"][indx] = self.exptime

        return observation_array


class FilterNexp(BandNexp):
    """Deprecated in favor of BandNexp"""

    def __init__(self, filtername="u", nexp=1, exptime=None):
        warnings.warn("Deprecated in favor of BandNexp", FutureWarning)
        super().__init__(bandname=filtername, nexp=nexp, exptime=exptime)


class TakeAsPairsDetailer(BaseDetailer):
    def __init__(self, bandname="r", exptime=None, nexp_dict=None, filtername=None):
        if filtername is not None:
            warnings.warn("filtername deprecated in favor of bandname", FutureWarning)
            bandname = filtername
        super(TakeAsPairsDetailer, self).__init__()
        self.bandname = bandname
        self.exptime = exptime
        self.nexp_dict = nexp_dict

    def __call__(self, observation_array, conditions):
        paired = copy.deepcopy(observation_array)
        if self.exptime is not None:
            paired["exptime"] = self.exptime
        paired["band"] = self.bandname
        if self.nexp_dict is not None:
            paired["nexp"] = self.nexp_dict[self.bandname]
        if conditions.current_band == self.bandname:
            tags = ["a", "b"]
        else:
            tags = ["b", "a"]

        paired["scheduler_note"] = np.char.add(paired["scheduler_note"], ", %s" % tags[0])
        observation_array["scheduler_note"] = np.char.add(
            observation_array["scheduler_note"], ", %s" % tags[1]
        )

        # Try to avoid extra filter changes and have "a" tag
        # come first.
        if conditions.current_band == self.bandname:
            result = np.concatenate([paired, observation_array])
        else:
            result = np.concatenate([observation_array, paired])

        return result


class TwilightTripleDetailer(BaseDetailer):
    def __init__(self, slew_estimate=5.0, n_repeat=3, update_note=True):
        super(TwilightTripleDetailer, self).__init__()
        self.slew_estimate = slew_estimate
        self.n_repeat = n_repeat
        self.update_note = update_note

    def __call__(self, obs_array, conditions):
        # Estimate how much time is left in the twilgiht block
        potential_times = np.array(
            [
                conditions.sun_n18_setting - conditions.mjd,
                conditions.sun_n12_rising - conditions.mjd,
            ]
        )

        potential_times = np.min(potential_times[np.where(potential_times > 0)]) * 24.0 * 3600.0

        # How long will observations take?
        cumulative_slew = np.arange(obs_array.size) * self.slew_estimate
        cumulative_expt = np.cumsum(obs_array["exptime"])
        cumulative_time = cumulative_slew + cumulative_expt
        # If we are way over, truncate the list before doing the triple
        if np.max(cumulative_time) > potential_times:
            max_indx = np.where(cumulative_time / self.n_repeat <= potential_times)[0]
            if np.size(max_indx) == 0:
                # Very bad magic number fudge
                max_indx = 3
            else:
                max_indx = np.max(max_indx)
                if max_indx == 0:
                    max_indx += 1
            obs_array = obs_array[0:max_indx]

        # Repeat the observations n times
        out_obs = []
        for i in range(self.n_repeat):
            sub_arr = copy.deepcopy(obs_array)
            if self.update_note:
                sub_arr["scheduler_note"] = np.char.add(sub_arr["scheduler_note"], ", %i" % i)
            out_obs.append(sub_arr)
        return np.concatenate(out_obs)


class TagRadialDetailer(BaseDetailer):
    """Update target_name and scheduler note based on visit location.

    Parameters
    ----------
    radius : `float`
        How close to a location on the sky a visit must be to
        get tagged. Defualr 5.6 (degrees).
    ra : `float`
        RA of the location. Default 0 (degrees).
    dec : `float`
        Dec of the location. Default -30 (degrees).
    note_append : `str`
        The string to append to the visit scheduler_note.
        Default ", deep area".
    target_prefix : `str`
        String to add to the start of the target_name field.
        Default "deep "
    target_name : `str`
        Target name to use. Overrides name generater with target_prefix.
        Default None.
    """

    def __init__(
        self, radius=5.6, ra=0, dec=-30.0, note_append=", deep area", target_prefix="deep ", target_name=None
    ):
        self.survey_features = {}
        self.radius = np.radians(radius)
        self.ra = np.radians(ra)
        self.dec = np.radians(dec)
        self.note_append = note_append
        self.target_name = target_prefix + f"{np.degrees(self.ra):.3f} {np.degrees(self.dec):.3f}"
        if target_name is not None:
            self.target_name = target_name

    def __call__(self, obsarray, conditions):
        distances = _angular_separation(obsarray["RA"], obsarray["dec"], self.ra, self.dec)
        if np.size(distances) == 1:
            distances = np.array([distances])
        in_region = np.where(distances <= self.radius)[0]
        obsarray["target_name"][in_region] = self.target_name
        obsarray["scheduler_note"][in_region] = np.char.add(
            obsarray["scheduler_note"][in_region], self.note_append
        )
        return obsarray


class LabelRegionDetailer(BaseDetailer):
    """Label which region(s) of the footprint we are in.

    Parameters
    ----------
    label_array : `np.array`
        A HEALpix array of strings that are labels for each HEALpix.
        Default of None uses CurrentAreaMap to load labels.
    field_for_label : `str`
        Which ObservationArray field should be modified.
        Default "target_name".
    camera : `str`
        Which camera model to use to convert a pointing to
        HEALpix IDs. Default "LSST".
    append : `bool`
        Should the labels be appended to `field_for_label`
        (True, default), or clobber (False).
    separator : `str`
        If append is True, what string to use when appending.
        Default ", ".
    """

    def __init__(
        self, label_array=None, field_for_label="target_name", camera="LSST", append=True, separator=", "
    ):
        if label_array is None:
            sky = CurrentAreaMap()
            _footprints, self.label_array = sky.return_maps()
        else:
            self.label_array = label_array
        self.field_for_label = field_for_label
        nside = hp.npix2nside(np.size(self.label_array))
        self.append = append
        self.separator = separator

        self.remove_vals = set(["", "", "None"])

        if camera == "LSST":
            self.pointing2hpindx = HpInLsstFov(nside=nside)
        else:
            raise ValueError("Unknown camera")

    def __call__(self, obs_array, conditions):
        for i in np.arange(obs_array.size):
            indx = self.pointing2hpindx(obs_array["RA"][i], obs_array["dec"][i])
            labels = np.unique(self.label_array[indx])
            # ignore things that are out of bounds
            labels = labels[np.where(labels != "")]
            # If there are no applicable labels, identify as OutOfFootprint
            if np.size(labels) == 0:
                result = set(["OOF"])
            else:
                result = set(labels)

            if self.append:
                prev_values = set(obs_array[self.field_for_label][i].split(self.separator))
                new_values = prev_values.union(result).difference(self.remove_vals)
            else:
                new_values = result.difference(self.remove_vals)
            obs_array[self.field_for_label][i] = self.separator.join(new_values)
        return obs_array


class LabelDDFDetailer(BaseDetailer):
    """Label if an observation is close enough to a
    DDF location to be tagged.

    Parameters
    ----------
    ddf_locations : `dict`
        Dictionary with keys of DDF names and values of RA,dec
        pairs (in degrees). Default of None uses ddf_locations
        utility to get default locations and names.
    field_for_label : `str`
        Which ObservationArray field should be modified.
        Default "target_name".
    append : `bool`
        Should the labels be appended to `field_for_label`
        (True, default), or clobber (False).
    match_radius : `float`
        The radius away an observation can be from a DDF center
        and still be tagged. Default 2.0 (degrees)
    separator : `str`
        If append is True, what string to use when appending.
        Default ", ".
    """

    def __init__(
        self, ddf_location=None, field_for_label="target_name", append=True, match_radius=2.0, separator=", "
    ):
        self.field_for_label = field_for_label
        self.append = append
        self.match_radius = np.radians(match_radius)
        self.separator = separator
        if ddf_location is None:
            self.ddf_locations = ddf_locations()
        else:
            self.ddf_locations = ddf_locations

        for key in self.ddf_locations:
            self.ddf_locations[key] = np.radians(self.ddf_locations[key])

        self.remove_vals = set(["", "", "None"])

    def __call__(self, obs_array, conditions):

        for name in self.ddf_locations:
            distances = _angular_separation(
                self.ddf_locations[name][0], self.ddf_locations[name][1], obs_array["RA"], obs_array["dec"]
            )
            distances = np.atleast_1d(distances)
            # Indexes of observations in obs_array which overlap the DDF
            indxes = np.where(distances <= self.match_radius)[0]

            name_set = set(["ddf_" + name.lower()])
            if indxes.size > 0:
                for indx in indxes:
                    if self.append:
                        prev_values = set(obs_array[self.field_for_label][indx].split(self.separator))
                        new_values = prev_values.union(name_set).difference(self.remove_vals)
                    else:
                        new_values = name_set.difference(self.remove_vals)
                    obs_array[self.field_for_label][indx] = self.separator.join(new_values)
        return obs_array


class LabelRegionsAndDDFs(BaseDetailer):
    """Run both LabelRegionDetailer and LabelDDFDetailer.

    Will always append onto current content of target_name.

    Parameters
    ----------
    label_array : `np.array`
        A HEALpix array of strings that are labels for each HEALpix.
        Default of None uses CurrentAreaMap to load labels.
    field_for_label : `str`
        Which ObservationArray field should be modified.
        Default "target_name".
    camera : `str`
        Which camera model to use to convert a pointing to
        HEALpix IDs. Default "LSST".
    ddf_locations : `dict`
        Dictionary with keys of DDF names and values of RA,dec
        pairs (in degrees). Default of None uses ddf_locations
        utility to get default locations and names.
    match_radius : `float`
        The radius away an observation can be from a DDF center
        and still be tagged. Default 2.0 (degrees)
    separator : `str`
        If append is True, what string to use when appending.
        Default ", ".
    """

    def __init__(
        self,
        label_array=None,
        field_for_label="target_name",
        camera="LSST",
        ddf_location=None,
        match_radius=2.0,
        separator=", ",
    ):
        self.label_detailer = LabelRegionDetailer(
            label_array=label_array,
            field_for_label=field_for_label,
            camera=camera,
            append=True,
            separator=separator,
        )
        self.ddf_detailer = LabelDDFDetailer(
            ddf_location=ddf_location,
            field_for_label=field_for_label,
            append=True,
            match_radius=match_radius,
            separator=separator,
        )

    def __call__(self, obs_array, conditions):

        result = self.label_detailer(obs_array, conditions)
        result = self.ddf_detailer(result, conditions)
        return result


class CopyValueDetailer(BaseDetailer):
    """Copy a value from one observation array column to another

    Parameters
    ----------
    source : `str`
        The name of the source column.
    destination : `str`
        Name of the destination column.
    """

    def __init__(self, source, destination):
        super().__init__()
        self.source = source
        self.destination = destination

    def __call__(self, obs_array, conditions):
        obs_array[self.destination] = obs_array[self.source]
        return obs_array
