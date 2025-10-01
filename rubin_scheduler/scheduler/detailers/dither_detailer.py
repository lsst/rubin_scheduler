__all__ = (
    "DitherDetailer",
    "EuclidDitherDetailer",
    "SplitDetailer",
    "CameraRotDetailer",
    "CameraSmallRotPerObservationListDetailer",
    "ComCamGridDitherDetailer",
    "DeltaCoordDitherDetailer",
)

import sys
import warnings

import numpy as np

from rubin_scheduler.scheduler.detailers import BaseDetailer
from rubin_scheduler.scheduler.features import Conditions, NObsCount
from rubin_scheduler.scheduler.utils import ObservationArray, rotx, thetaphi2xyz, wrap_ra_dec, xyz2thetaphi
from rubin_scheduler.utils import (
    _approx_altaz2pa,
    _approx_ra_dec2_alt_az,
    bearing,
    ddf_locations,
    dest_latlon,
    gnomonic_project_tosky,
    rotation_converter,
)


class DitherDetailer(BaseDetailer):
    """
    make a uniform dither pattern. Offset by a maximum radius in a random
    direction. Mostly intended for DDF pointings, the BaseMarkovDF_survey
    class includes dithering for large areas.

    Parameters
    ----------
    max_dither : `float` (0.7)
        The maximum dither size to use (degrees).
    per_night : `bool` (True)
        If true, use the same dither offset for an entire night. If False,
        generate a dither position per visit.
    n_in_night : `int` (1200)
        The number of unique positions to generate in a night.
        Only used if per_night is False.
    big_int : `int`
        A big integer to use for seeding random number generator
        in case one send in a `night` value that is negative.
    """

    def __init__(
        self, max_dither=0.7, per_night=True, n_in_night=1200, nnights=None, big_int=700000, seed=None
    ):

        if nnights is not None:
            warnings.warn("kwarg nnights deprecated", FutureWarning)
        if seed is not None:
            warnings.warn("kwarg seed deprecated", FutureWarning)

        # Check we can run for 10 years no problem
        test = 36525 < big_int < (sys.maxsize - 36525)
        if not test:
            raise ValueError("Value used for big_int won't work for 10 year survey.")

        self.survey_features = {"n_in_night": NObsCount(per_night=True)}

        self.current_night = -np.inf
        self.max_dither = np.radians(max_dither)
        self.n_in_night = n_in_night
        self.per_night = per_night

        self.offset_ra = None
        self.offset_dec = None

        self.current_night = -np.inf
        self.big_int = big_int

    def _new_ang_rad(self, night):

        # Seed with the current night
        if night < 0:
            seed = self.big_int + night
        else:
            seed = night
        rng = np.random.default_rng(seed)
        if self.per_night:
            n_gen = 1
        else:
            n_gen = self.n_in_night

        self.angles = rng.random(n_gen) * 2 * np.pi
        self.radii = self.max_dither * np.sqrt(rng.random(n_gen))

    def _generate_offsets(self, n_offsets, night):

        if self.per_night:
            if night != self.current_night:
                self._new_ang_rad(night)
                self.current_night = night
                angle = self.angles
                radius = self.radii

                self.offsets_ra = radius * np.cos(angle)
                self.offsets_dec = radius * np.sin(angle)
        else:
            if night != self.current_night:
                self._new_ang_rad(night)
                self.current_night = night
            indx_in_night = self.survey_features["n_in_night"].feature
            angle = self.angles[indx_in_night : indx_in_night + n_offsets]
            radius = self.radii[indx_in_night : indx_in_night + n_offsets]
            self.offsets_ra = radius * np.cos(angle)
            self.offsets_dec = radius * np.sin(angle)

        return self.offsets_ra, self.offsets_dec

    def __call__(self, obs_array, conditions):
        if len(obs_array) == 0:
            return obs_array

        # Generate offsets in RA and Dec
        offsets_ra, offsets_dec = self._generate_offsets(len(obs_array), conditions.night)
        new_ra, new_dec = gnomonic_project_tosky(offsets_ra, offsets_dec, obs_array["RA"], obs_array["dec"])

        obs_array["RA"] = new_ra
        obs_array["dec"] = new_dec

        return obs_array


class DeltaCoordDitherDetailer(BaseDetailer):
    """Dither pattern set by user. Each input observation is
    expanded to have an observation at each dither position.

    Parameters
    ----------
    delta_ra : `np.array`
        Angular distances to move on sky in the RA-direction (degrees).
    delta_dec : `np.array`
        Angular distances to move on sky in the dec-direction (degree).
    delta_rotskypos : `np.array`
        Angular shifts to make in the rotskypos values (degrees). Default None
        applies no rotational shift
    """

    def __init__(self, delta_ra, delta_dec, delta_rotskypos=None):

        if delta_rotskypos is not None:
            if np.size(delta_ra) != np.size(delta_rotskypos):
                raise ValueError(
                    "size of delta_ra (%i) is not equal to size of delta_rotskypos (%i)"
                    % (np.size(delta_ra), np.size(delta_rotskypos))
                )
        if np.size(delta_ra) != np.size(delta_dec):
            raise ValueError(
                "Sizes of delta_ra (%i) and delta_dec (%i) do not match."
                % (np.size(delta_ra), np.size(delta_dec))
            )

        self.survey_features = {}
        self.delta_ra = np.radians(delta_ra)
        self.delta_dec = np.radians(delta_dec)
        if delta_rotskypos is None:
            self.delta_rotskypos = delta_rotskypos
        else:
            self.delta_rotskypos = np.radians(delta_rotskypos)

    def __call__(self, obs_array, conditions):

        output_array_list = []
        for obs in obs_array:
            dithered_observations = ObservationArray(self.delta_ra.size)
            for key in obs.dtype.names:
                dithered_observations[key] = obs[key]

            # Generate grid on the equator (so RA offsets are not small)
            new_decs = self.delta_dec.copy()
            # Adding 90 deg here for later rotation about x-axis
            new_ras = self.delta_ra + np.pi / 2

            # Rotate ra and dec about the x-axis
            # pi/2 term to convert dec to phi
            x, y, z = thetaphi2xyz(new_ras, new_decs + np.pi / 2.0)
            # rotate so offsets are now centered on proper dec
            xp, yp, zp = rotx(obs["dec"], x, y, z)
            theta, phi = xyz2thetaphi(xp, yp, zp)

            new_decs = phi - np.pi / 2
            # Remove 90 degree rot from earlier and center on proper RA
            new_ras = theta - np.pi / 2 + obs["RA"]

            # Make sure coords are in proper range
            new_ras, new_decs = wrap_ra_dec(new_ras, new_decs)

            dithered_observations["RA"] = new_ras
            dithered_observations["dec"] = new_decs
            if self.delta_rotskypos is not None:
                dithered_observations["rotSkyPos_desired"] += self.delta_rotskypos
            output_array_list.append(dithered_observations)

        result = np.concatenate(output_array_list)
        return result


class EuclidDitherDetailer(BaseDetailer):
    """Directional dithering for Euclid DDFs

    Parameters
    ----------
    dither_bearing_dir : `list`
        A list with the dither amplitude in along the axis
        connecting the two positions. Default [-0.25, 1] in degrees.
    dither_bearing_perp : `list`
        A list with the dither amplitude perpendicular to the
        axis connecting the two positions. Default [-0.25, 1]
        in degrees.
    seed : `float`
        Random number seed to use (42).
    per_night : `bool`
        If dither shifts should be per night (default True),
        or if dithers should be every pointing (False).
    ra_a, dec_a, ra_b, dec_b : `float`
        Positions for the two field centers. Default None
        will load the positions from
        rubin_scheduler.utils.ddf_locations
    n_in_night : `int`
        Number of positions to generate dither positions for
        within a night.
    """

    def __init__(
        self,
        dither_bearing_dir=[-0.25, 1],
        dither_bearing_perp=[-0.25, 0.25],
        per_night=True,
        ra_a=None,
        dec_a=None,
        ra_b=None,
        dec_b=None,
        n_in_night=1200,
        nnights=None,
        seed=None,
    ):
        if nnights is not None:
            warnings.warn("kwarg nnights deprecated", FutureWarning)
        if seed is not None:
            warnings.warn("kwarg seed deprecated", FutureWarning)
        self.n_in_night = n_in_night
        self.survey_features = {"n_in_night": NObsCount(per_night=True)}

        default_locations = ddf_locations()

        if ra_a is None:
            self.ra_a = np.radians(default_locations["EDFS_a"][0])
        else:
            self.ra_a = np.radians(ra_a)
        if ra_b is None:
            self.ra_b = np.radians(default_locations["EDFS_b"][0])
        else:
            self.ra_b = np.radians(ra_b)
        if dec_a is None:
            self.dec_a = np.radians(default_locations["EDFS_a"][1])
        else:
            self.dec_a = np.radians(dec_a)
        if dec_b is None:
            self.dec_b = np.radians(default_locations["EDFS_b"][1])
        else:
            self.dec_b = np.radians(dec_b)

        self.dither_bearing_dir = np.radians(dither_bearing_dir)
        self.dither_bearing_perp = np.radians(dither_bearing_perp)

        self.bearing_atob = bearing(self.ra_a, self.dec_a, self.ra_b, self.dec_b)
        self.bearing_btoa = bearing(self.ra_b, self.dec_b, self.ra_a, self.dec_a)

        self.current_night = -np.inf

        self.per_night = per_night
        self.shifted_ra_a = None
        self.shifted_dec_a = None
        self.shifted_ra_b = None
        self.shifted_dec_b = None

    def _new_ang_rad(self, night):

        # seed with the current night
        rng = np.random.default_rng(night)
        if self.per_night:
            n_gen = 1
        else:
            n_gen = self.n_in_night

        self.bearings_mag_1 = rng.uniform(
            low=self.dither_bearing_dir.min(),
            high=self.dither_bearing_dir.max(),
            size=n_gen,
        )
        self.perp_mag_1 = rng.uniform(
            low=self.dither_bearing_perp.min(),
            high=self.dither_bearing_perp.max(),
            size=n_gen,
        )

        self.bearings_mag_2 = rng.uniform(
            low=self.dither_bearing_dir.min(),
            high=self.dither_bearing_dir.max(),
            size=n_gen,
        )
        self.perp_mag_2 = rng.uniform(
            low=self.dither_bearing_perp.min(),
            high=self.dither_bearing_perp.max(),
            size=n_gen,
        )

    def _generate_offsets(self, n_offsets, night):

        indx_in_night = self.survey_features["n_in_night"].feature
        if night != self.current_night:
            self.indx_in_night = 0
            self._new_ang_rad(night)
            self.current_night = night
        if self.per_night:
            bearing_mag = self.bearings_mag_1
            perp_mag = self.perp_mag_1
        else:
            bearing_mag = self.bearings_mag_1[indx_in_night : indx_in_night + n_offsets]
            perp_mag = self.perp_mag_1[indx_in_night : indx_in_night + n_offsets]

        # Move point a along the bearings
        self.shifted_dec_a, self.shifted_ra_a = dest_latlon(
            bearing_mag, self.bearing_atob, self.dec_a, self.ra_a
        )
        self.shifted_dec_a, self.shifted_ra_a = dest_latlon(
            perp_mag,
            self.bearing_atob + np.pi / 2.0,
            self.shifted_dec_a,
            self.shifted_ra_a,
        )

        # Shift the second position
        if self.per_night:
            bearing_mag = self.bearings_mag_2
            perp_mag = self.perp_mag_2
        else:
            bearing_mag = self.bearings_mag_2[indx_in_night : indx_in_night + n_offsets]
            perp_mag = self.perp_mag_2[indx_in_night : indx_in_night + n_offsets]

        self.shifted_dec_b, self.shifted_ra_b = dest_latlon(
            bearing_mag, self.bearing_btoa, self.dec_b, self.ra_b
        )
        self.shifted_dec_b, self.shifted_ra_b = dest_latlon(
            perp_mag,
            self.bearing_btoa + np.pi / 2.0,
            self.shifted_dec_b,
            self.shifted_ra_b,
        )

        return (
            self.shifted_ra_a,
            self.shifted_dec_a,
            self.shifted_ra_b,
            self.shifted_dec_b,
        )

    def __call__(self, obs_array, conditions):
        if np.size(obs_array) == 0:
            return obs_array

        # Generate offsets in RA and Dec
        ra_a, dec_a, ra_b, dec_b = self._generate_offsets(len(obs_array), conditions.night)

        ra = []
        dec = []
        for i, obs in enumerate(obs_array):
            if "DD:EDFS_a" in obs["scheduler_note"][0:9]:
                ra.append(ra_a)
                dec.append(dec_a)
            elif "DD:EDFS_b" in obs["scheduler_note"][0:9]:
                ra.append(ra_b)
                dec.append(dec_b)
            else:
                raise ValueError("scheduler_note does not contain EDFS_a or EDFS_b.")
        obs_array["RA"] = np.concatenate(ra)
        obs_array["dec"] = np.concatenate(dec)
        return obs_array


class SplitDetailer(BaseDetailer):
    """Combine two detailers, but choose which one of them to use, based on
    the presence of a specified string in the scheduler_note.

    Useful to identify different kinds of observations in a ScriptedSurvey
    and then apply different detailers (such as EuclidDetailer vs. standard
    DDF dither detailer).

    Parameters
    ----------
    det1 : `detailers.BaseDetailer`
        The first detailer, to use if the `split_str` is not present
        in scheduler_note.
    det2 : `detailers.BaseDetailer`
        The second detailer, to use if `split_str` is present.
    split_str : `str`
        Search for this string in scheduler_note (matches sub-string).
    """

    def __init__(
        self,
        det1: BaseDetailer,
        det2: BaseDetailer,
        split_str: str = "EDFS",
    ):
        self.det1 = det1
        self.det2 = det2
        self.split_str = split_str

    def __call__(self, observation_array: ObservationArray, conditions: Conditions) -> ObservationArray:
        string_in = [self.split_str in note for note in observation_array["scheduler_note"]]
        string_out = np.logical_not(string_in)

        observation_array[string_out] = self.det1(observation_array[string_out], conditions)
        observation_array[string_in] = self.det2(observation_array[string_in], conditions)
        return observation_array


class CameraRotDetailer(BaseDetailer):
    """
    Randomly set the camera rotation, either for each exposure,
    for each time the detailer is called, or per night.

    Note dithering for each exposure may not be repeatable if
    scheduler is restarted.

    Parameters
    ----------
    max_rot : `float` (90.)
        The maximum amount to offset the camera (degrees)
    min_rot : `float` (90)
        The minimum to offset the camera (degrees)
    dither : `str`
        If "night", change positions per night. If call, change per call.
        If "all", randomize per visit. Default "night".
    telescope : `str`
        Telescope name. Options of "rubin" or "auxtel". Default "rubin".
    n_in_night : `int`
        Number of dither positions to generate. Should be greater than
        the maximum number of visits expected in a night (if
        dither set to "call"), or greather than number of ngihts
        in survey if dither set to "night". Default 7350.
    """

    def __init__(
        self,
        max_rot=90.0,
        min_rot=-90.0,
        dither="night",
        per_night=None,
        seed=42,
        nnights=None,
        telescope="rubin",
        n_in_night=7350,
    ):
        self.survey_features = {}

        if nnights is not None:
            warnings.warn("nnights deprecated", FutureWarning)

        if per_night is True:
            warnings.warn("per_night deprecated, setting dither='night'", FutureWarning)
            dither = "night"
        if per_night is False:
            warnings.warn("per_night deprecated, setting dither='all'", FutureWarning)
            dither = "all"

        if dither is True:
            warnings.warn("dither=True deprecated, swapping to dither='night'", FutureWarning)
            dither = "night"

        if dither is False:
            warnings.warn("dither=False deprecated, swapping to dither='all'", FutureWarning)
            dither = "all"

        self.current_night = -np.inf
        self.max_rot = np.radians(max_rot)
        self.min_rot = np.radians(min_rot)
        self.range = self.max_rot - self.min_rot
        self.dither = dither
        self.rng = np.random.default_rng(seed)
        self.n_in_night = n_in_night
        if dither == "call":
            self.offsets = None
        else:
            self.offsets = self.rng.random(n_in_night)

        self.offset = None
        self.rc = rotation_converter(telescope=telescope)

        self.call_num = 0

    def _generate_offsets(self, n_offsets, night):
        if self.dither == "night":
            if night != self.current_night:
                self.current_night = night
                self.offset = self.offsets[night] * self.range + self.min_rot
            offsets = np.ones(n_offsets) * self.offset
        elif self.dither == "call":
            if night != self.current_night:
                self.current_night = night
                self.call_num = 0
                self.rng = np.random.default_rng(night)
                self.offsets = self.rng.random(self.n_in_night)
            offsets = self.offsets[self.call_num] * self.range + self.min_rot
            self.call_num += 1
            offsets = np.ones(n_offsets) * offsets
        elif self.dither == "all":
            self.rng = np.random.default_rng()
            offsets = self.rng.random(n_offsets) * self.range + self.min_rot
        else:
            raise ValueError("dither kwarg must be set to 'night', 'call', or 'all'.")

        return offsets

    def __call__(self, observation_array, conditions):
        # Generate offsets in camamera rotator
        offsets = self._generate_offsets(len(observation_array), conditions.night)

        alt, az = _approx_ra_dec2_alt_az(
            observation_array["RA"],
            observation_array["dec"],
            conditions.site.latitude_rad,
            conditions.site.longitude_rad,
            conditions.mjd,
        )
        obs_pa = _approx_altaz2pa(alt, az, conditions.site.latitude_rad)
        observation_array["rotSkyPos"] = self.rc._rottelpos2rotskypos(offsets, obs_pa)
        observation_array["rotTelPos"] = offsets

        return observation_array


class CameraSmallRotPerObservationListDetailer(BaseDetailer):
    """
    Randomly set the camera rotation for each observation list.

    Generates a small sequential offset for sequential visits
    in the same band; adds a random offset for each band change.

    Parameters
    ----------
    max_rot : `float`, optional
        The maximum amount to offset the camera (degrees).
        Default of 85 allows some padding for camera rotator.
    min_rot : `float`, optional
        The minimum to offset the camera (degrees)
        Default of -85 allows some padding for camera rotator.
    seed : `int`, optional
        Seed for random number generation (per night).
    per_visit_rot : `float`, optional
        Sequential rotation to add per visit.
    telescope : `str`, optional
        Telescope name. Options of "rubin" or "auxtel". Default "rubin".
        This is used to determine conversions between rotSkyPos and rotTelPos.
    """

    def __init__(self, max_rot=85.0, min_rot=-85.0, seed=42, per_visit_rot=0.0, telescope="rubin"):
        self.survey_features = {}

        self.current_night = -1
        self.max_rot = np.radians(max_rot)
        self.min_rot = np.radians(min_rot)
        self.rot_range = self.max_rot - self.min_rot
        self.seed = seed
        self.per_visit_rot = np.radians(per_visit_rot)
        self.offset = None
        self.rc = rotation_converter(telescope=telescope)

    def _generate_offsets_band_change(self, band_list, mjd, initial_offset):
        """Generate a random camera rotation for each band change
        or add a small offset for each sequential observation.
        """
        mjd_hash = round(100 * (np.asarray(mjd).item() % 100))
        rng = np.random.default_rng(mjd_hash * self.seed)

        offsets = np.zeros(len(band_list))

        # Find the locations of the band changes
        band_changes = np.where(np.array(band_list[:-1]) != np.array(band_list[1:]))[0]
        band_changes = np.concatenate([np.array([-1]), band_changes])
        # But add one because of counting and offsets above.
        band_changes += 1
        # Count visits per band in the sequence.
        nvis_per_band = np.concatenate(
            [np.diff(band_changes), np.array([len(band_list) - 1 - band_changes[-1]])]
        )
        # Set up the random rotator offsets for each band change
        # This includes first rotation .. maybe not needed?
        for fchange_idx, nvis_f in zip(band_changes, nvis_per_band):
            rot_range = self.rot_range - self.per_visit_rot * nvis_f
            # At the band change spot, update to random offset
            offsets[fchange_idx:] = rng.random() * rot_range + self.min_rot
            # After the band change point, add incremental rotation
            # (we'll wipe this when we get to next fchange_idx)
            offsets[fchange_idx:] += self.per_visit_rot * np.arange(len(band_list) - fchange_idx)

        offsets = np.where(offsets > self.max_rot, self.max_rot, offsets)
        return offsets

    def __call__(self, observation_list, conditions):
        # Generate offsets in camera rotator
        band_list = [np.asarray(obs["band"]).item() for obs in observation_list]
        offsets = self._generate_offsets_band_change(band_list, conditions.mjd, conditions.rot_tel_pos)

        for i, obs in enumerate(observation_list):
            alt, az = _approx_ra_dec2_alt_az(
                obs["RA"],
                obs["dec"],
                conditions.site.latitude_rad,
                conditions.site.longitude_rad,
                conditions.mjd,
            )
            obs_pa = _approx_altaz2pa(alt, az, conditions.site.latitude_rad)
            obs["rotSkyPos"] = self.rc._rottelpos2rotskypos(offsets[i], obs_pa)
            obs["rotTelPos"] = offsets[i]

        return observation_list


class ComCamGridDitherDetailer(BaseDetailer):
    """
    Generate an offset pattern to synthesize a 2x2 grid of ComCam pointings.

    Parameters
    ----------
    rotTelPosDesired : `float`, (0.)
        The physical rotation angle of the camera rotator (degrees)
    scale : `float` (0.355)
        Half of the offset between grid pointing centers. (degrees)
    dither : `float` (0.05)
        Dither offsets within grid to fill chip gaps. (degrees)
    telescope : `str`, ("comcam")
        Telescope name. Default "comcam".
        This is used to determine conversions between rotSkyPos and rotTelPos.
    """

    def __init__(self, rotTelPosDesired=0.0, scale=0.355, dither=0.05, telescope="comcam"):
        self.survey_features = {}

        self.rotTelPosDesired = np.radians(rotTelPosDesired)
        self.scale = np.radians(scale)
        self.dither = np.radians(dither)
        self.rc = rotation_converter(telescope=telescope)

    def _rotate(self, x, y, angle):
        x_rot = x * np.cos(angle) - y * np.sin(angle)
        y_rot = x * np.sin(angle) + y * np.cos(angle)
        return x_rot, y_rot

    def _generate_offsets(self, n_offsets, band_list, rotSkyPos):
        # 2 x 2 pointing grid
        x_grid = np.array([-1.0 * self.scale, -1.0 * self.scale, self.scale, self.scale])
        y_grid = np.array([-1.0 * self.scale, self.scale, self.scale, -1.0 * self.scale])
        x_grid_rot, y_grid_rot = self._rotate(x_grid, y_grid, -1.0 * rotSkyPos)
        offsets_grid_rot = np.array([x_grid_rot, y_grid_rot]).T

        # Dither pattern within grid to fill chip gaps
        # Psuedo-random offsets
        x_dither = np.array(
            [
                0.0,
                -0.5 * self.dither,
                -1.25 * self.dither,
                1.5 * self.dither,
                0.75 * self.dither,
            ]
        )
        y_dither = np.array(
            [
                0.0,
                -0.75 * self.dither,
                1.5 * self.dither,
                1.25 * self.dither,
                -0.5 * self.dither,
            ]
        )
        x_dither_rot, y_dither_rot = self._rotate(x_dither, y_dither, -1.0 * rotSkyPos)
        offsets_dither_rot = np.array([x_dither_rot, y_dither_rot]).T

        # Find the indices of the band changes
        band_changes = np.where(np.array(band_list[:-1]) != np.array(band_list[1:]))[0]
        band_changes = np.concatenate([np.array([-1]), band_changes])
        band_changes += 1

        offsets = []
        index_band = 0
        for ii in range(0, n_offsets):
            if ii in band_changes:
                # Reset the count after each band change
                index_band = 0

            index_grid = index_band % 4
            index_dither = np.floor(index_band / 4).astype(int) % 5
            offsets.append(offsets_grid_rot[index_grid] + offsets_dither_rot[index_dither])
            index_band += 1

        return np.vstack(offsets)

    def __call__(self, observation_list, conditions):
        if len(observation_list) == 0:
            return observation_list

        band_list = [np.asarray(obs["band"]).item() for obs in observation_list]

        # Initial estimate of rotSkyPos corresponding to desired rotTelPos
        alt, az, pa = _approx_ra_dec2_alt_az(
            observation_list[0]["RA"],
            observation_list[0]["dec"],
            conditions.site.latitude_rad,
            conditions.site.longitude_rad,
            conditions.mjd,
            return_pa=True,
        )
        rotSkyPos = self.rc._rottelpos2rotskypos(self.rotTelPosDesired, pa)

        # Generate offsets in RA and Dec
        offsets = self._generate_offsets(len(observation_list), band_list, rotSkyPos)

        # Project offsets onto sky
        obs_array = np.concatenate(observation_list)
        new_ra, new_dec = gnomonic_project_tosky(
            offsets[:, 0], offsets[:, 1], obs_array["RA"], obs_array["dec"]
        )
        new_ra, new_dec = wrap_ra_dec(new_ra, new_dec)

        # Update observations
        for ii in range(0, len(observation_list)):
            observation_list[ii]["RA"] = new_ra[ii]
            observation_list[ii]["dec"] = new_dec[ii]

            alt, az, pa = _approx_ra_dec2_alt_az(
                new_ra[ii],
                new_dec[ii],
                conditions.site.latitude_rad,
                conditions.site.longitude_rad,
                conditions.mjd,
                return_pa=True,
            )
            observation_list[ii]["rotSkyPos"] = rotSkyPos
            observation_list[ii]["rotTelPos"] = self.rc._rotskypos2rottelpos(rotSkyPos, pa)

        return observation_list
