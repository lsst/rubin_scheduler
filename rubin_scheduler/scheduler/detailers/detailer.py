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
    "FixedSkyAngleDetailer",
    "ParallacticRotationDetailer",
    "FlushByDetailer",
    "RandomFilterDetailer",
    "TrackingInfoDetailer",
    "AltAz2RaDecDetailer",
)

import copy

import numpy as np

from rubin_scheduler.scheduler.utils import IntRounded
from rubin_scheduler.utils import (
    DEFAULT_NSIDE,
    _angular_separation,
    _approx_alt_az2_ra_dec,
    _approx_altaz2pa,
    _approx_ra_dec2_alt_az,
    pseudo_parallactic_angle,
    rotation_converter,
)


class BaseDetailer:
    """
    A Detailer is an object that takes a list of proposed observations and
    adds "details" to them. The primary purpose is that the Markov Decision
    Process does an excelent job selecting RA,Dec,filter combinations, but we
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
        List of observations.
        """

        return observation_list


class TrackingInfoDetailer(BaseDetailer):
    """Fill in lots of the different tracking strings for an observation."""

    def __init__(self, target_name=None, science_program=None, observation_reason=None):
        self.survey_features = {}
        self.target_name = target_name
        self.science_program = science_program
        self.observation_reason = observation_reason

    def __call__(self, observation_list, conditions):
        for obs in observation_list:
            if self.science_program is not None:
                obs["science_program"] = self.science_program
            if self.target_name is not None:
                obs["target_name"] = self.target_name
            if self.observation_reason is not None:
                obs["observation_reason"] = self.observation_reason

        return observation_list


class FlushByDetailer(BaseDetailer):
    """Set the MJD an observation should be flushed from the scheduler
    queue if not yet completed.

    Parameters
    ----------
    flush_time : float
        The time to flush after the current MJD. Default 60 minutes
    """

    def __init__(self, flush_time=60, nside=DEFAULT_NSIDE):
        self.survey_features = {}
        self.nside = nside
        self.flush_time = flush_time / 60 / 24.0

    def __call__(self, observation_list, conditions):
        for obs in observation_list:
            obs["flush_by_mjd"] = conditions.mjd + self.flush_time
        return observation_list


class AltAz2RaDecDetailer(BaseDetailer):
    """Set RA,dec for an observation that only has alt,az"""

    def __call__(self, observation_list, conditions):
        for observation in observation_list:
            ra, dec = _approx_alt_az2_ra_dec(
                observation["alt"],
                observation["az"],
                conditions.site.latitude_rad,
                conditions.site.longitude_rad,
                conditions.mjd,
            )
            observation["RA"] = ra
            observation["dec"] = dec

        return observation_list


class RandomFilterDetailer(BaseDetailer):
    """Pick a random filter for the observations

    Parameters
    ----------
    filters : `str`
        The filters to randomize. Default 'riz'
    nights_to_prep : `int`
        The number of nights to generate random filters for.
        Default 10000.
    seed : number
        Seed for RNG. Defaut 42
    fallback_order : `str`
        If the desired filter is not mounted, goes through
        `fallback_order` and uses the first filter that is
        available
    """

    def __init__(self, filters="riz", nights_to_prep=10000, seed=42, fallback_order="rizgyu"):
        self.survey_features = []
        self.fallback_order = fallback_order

        rng = np.random.default_rng(seed)
        self.night2filter_int = rng.integers(low=0, high=len(filters), size=nights_to_prep)

        self.filter_dict = {}
        for i, filtername in enumerate(filters):
            self.filter_dict[i] = filtername

    def __call__(self, observation_list, conditions):

        filter_to_use = self.filter_dict[self.night2filter_int[conditions.night]]
        # Filter not available
        if filter_to_use not in conditions.mounted_filters:
            is_mounted = [filtername in conditions.mounted_filters for filtername in self.fallback_order]
            indx = np.min(np.where(is_mounted))
            filter_to_use = self.fallback_order[indx]

        for obs in observation_list:
            obs["filter"] = filter_to_use
        return observation_list


class ParallacticRotationDetailer(BaseDetailer):
    """Set the rotator to near the parallactic angle"""

    def __init__(self, telescope="rubin"):
        self.rc = rotation_converter(telescope=telescope)
        self.survey_features = {}

    def __call__(self, observation_list, conditions, limits=[-270, 270]):
        limits = np.radians(limits)
        for obs in observation_list:
            alt, az = _approx_ra_dec2_alt_az(
                obs["RA"],
                obs["dec"],
                conditions.site.latitude_rad,
                conditions.site.longitude_rad,
                conditions.mjd,
            )
            obs_pa = _approx_altaz2pa(alt, az, conditions.site.latitude_rad)
            obs["rotSkyPos_desired"] = obs_pa

            resulting_rot_tel_pos = self.rc._rotskypos2rottelpos(obs["rotSkyPos_desired"], obs_pa)

            if resulting_rot_tel_pos > np.max(limits):
                resulting_rot_tel_pos -= 2 * np.pi
            if resulting_rot_tel_pos < np.min(limits):
                resulting_rot_tel_pos += 2 * np.pi

            # If those corrections still leave us bad, just pull it back 180.
            if resulting_rot_tel_pos > np.max(limits):
                resulting_rot_tel_pos -= np.pi

            # The rotTelPos overides everything else.
            obs["rotTelPos"] = resulting_rot_tel_pos
            # if the rotSkyPos_desired isn't possible, fall back to this.
            obs["rotTelPos_backup"] = 0

        return observation_list


class Rottep2RotspDesiredDetailer(BaseDetailer):
    """Convert all the rotTelPos values to rotSkyPos_desired"""

    def __init__(self, telescope="rubin"):
        self.rc = rotation_converter(telescope=telescope)
        self.survey_features = {}

    def __call__(self, observation_list, conditions):
        obs_array = np.concatenate(observation_list)

        alt, az = _approx_ra_dec2_alt_az(
            obs_array["RA"],
            obs_array["dec"],
            conditions.site.latitude_rad,
            conditions.site.longitude_rad,
            conditions.mjd,
        )
        obs_pa = _approx_altaz2pa(alt, az, conditions.site.latitude_rad)

        rot_sky_pos_desired = self.rc._rotskypos2rottelpos(obs_array["rotTelPos"], obs_pa)

        for obs, rotsp_d in zip(observation_list, rot_sky_pos_desired):
            obs["rotTelPos_backup"] = obs["rotTelPos"] + 0
            obs["rotTelPos"] = np.nan
            obs["rotSkyPos"] = np.nan
            obs["rotSkyPos_desired"] = rotsp_d

        return observation_list


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

    def __call__(self, observation_list, conditions):
        # XXX--should I convert the list into an array and get rid of this
        # loop?
        for obs in observation_list:
            obs_pa, alt, az = pseudo_parallactic_angle(
                obs["RA"],
                obs["dec"],
                conditions.mjd,
                np.degrees(conditions.site.longitude_rad),
                np.degrees(conditions.site.latitude_rad),
            )

            obs["rotSkyPos"] = self.rc.rottelpos2rotskypos(0.0, obs_pa)

        return observation_list


class Comcam90rotDetailer(BaseDetailer):
    """
    Detailer to set the camera rotation so rotSkyPos is 0, 90, 180, or
    270 degrees. Whatever is closest to rotTelPos of zero.
    """

    def __init__(self, telescope="rubin", nside=DEFAULT_NSIDE):
        self.rc = rotation_converter(telescope=telescope)
        self.survey_features = {}

    def __call__(self, observation_list, conditions):
        favored_rot_sky_pos = np.radians([0.0, 90.0, 180.0, 270.0, 360.0]).reshape(5, 1)
        obs_array = np.concatenate(observation_list)

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

        ang_diff = np.abs(self.rc._rotskypos2rottelpos(favored_rot_sky_pos, parallactic_angle))
        min_indxs = np.argmin(ang_diff, axis=0)
        # can swap 360 and zero if needed?
        final_rot_sky_pos = favored_rot_sky_pos[min_indxs]
        # Set all the observations to the proper rotSkyPos
        for rsp, obs in zip(final_rot_sky_pos, observation_list):
            obs["rotSkyPos"] = rsp

        return observation_list


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

    def __call__(self, observation_list, conditions):
        for observation in observation_list:
            observation["rotSkyPos"] = self.sky_angle

        return observation_list


class CloseAltDetailer(BaseDetailer):
    """
    re-order a list of observations so that the closest in altitude to
    the current pointing is first.

    Parameters
    ----------
    alt_band : `float` (10)
        The altitude band to try and stay in (degrees)
    """

    def __init__(self, alt_band=10.0):
        super(CloseAltDetailer, self).__init__()
        self.alt_band = IntRounded(np.radians(alt_band))

    def __call__(self, observation_list, conditions):
        obs_array = np.concatenate(observation_list)
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
        result = observation_list[indx:] + observation_list[:indx]
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

    def __call__(self, observation_list, conditions):
        if np.size(conditions.scheduled_observations) > 0:
            new_flush = np.min(conditions.scheduled_observations) - self.tol
            for obs in observation_list:
                if obs["flush_by_mjd"] > new_flush:
                    obs["flush_by_mjd"] = new_flush
        return observation_list


class FilterNexp(BaseDetailer):
    """Demand one filter always be taken as a certain number of exposures"""

    def __init__(self, filtername="u", nexp=1, exptime=None):
        super(FilterNexp, self).__init__()
        self.filtername = filtername
        self.nexp = nexp
        self.exptime = exptime

    def __call__(self, observation_list, conditions):
        for obs in observation_list:
            if obs["filter"] == self.filtername:
                obs["nexp"] = self.nexp
                if self.exptime is not None:
                    obs["exptime"] = self.exptime
        return observation_list


class TakeAsPairsDetailer(BaseDetailer):
    def __init__(self, filtername="r", exptime=None, nexp_dict=None):
        """"""
        super(TakeAsPairsDetailer, self).__init__()
        self.filtername = filtername
        self.exptime = exptime
        self.nexp_dict = nexp_dict

    def __call__(self, observation_list, conditions):
        paired = copy.deepcopy(observation_list)
        if self.exptime is not None:
            for obs in paired:
                obs["exptime"] = self.exptime
        for obs in paired:
            obs["filter"] = self.filtername
            if self.nexp_dict is not None:
                obs["nexp"] = self.nexp_dict[self.filtername]
        if conditions.current_filter == self.filtername:
            for obs in paired:
                obs["scheduler_note"] = obs["scheduler_note"][0] + ", a"
            for obs in observation_list:
                obs["scheduler_note"] = obs["scheduler_note"][0] + ", b"
            result = paired + observation_list
        else:
            for obs in paired:
                obs["scheduler_note"] = obs["scheduler_note"][0] + ", b"
            for obs in observation_list:
                obs["scheduler_note"] = obs["scheduler_note"][0] + ", a"
            result = observation_list + paired
        return result


class TwilightTripleDetailer(BaseDetailer):
    def __init__(self, slew_estimate=5.0, n_repeat=3, update_note=True):
        super(TwilightTripleDetailer, self).__init__()
        self.slew_estimate = slew_estimate
        self.n_repeat = n_repeat
        self.update_note = update_note

    def __call__(self, observation_list, conditions):
        obs_array = np.concatenate(observation_list)

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
            observation_list = observation_list[0:max_indx]

        # Repeat the observations n times
        out_obs = []
        for i in range(self.n_repeat):
            sub_list = copy.deepcopy(observation_list)
            if self.update_note:
                for obs in sub_list:
                    obs["scheduler_note"][0] += ", %i" % i
            out_obs.extend(sub_list)
        return out_obs
