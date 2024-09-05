__all__ = (
    "DitherDetailer",
    "EuclidDitherDetailer",
    "CameraRotDetailer",
    "CameraSmallRotPerObservationListDetailer",
)

import numpy as np

from rubin_scheduler.scheduler.detailers import BaseDetailer
from rubin_scheduler.scheduler.utils import wrap_ra_dec
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
        If true, us the same dither offset for an entire night
    nnights : `int` (7305)
        The number of nights to pre-generate random dithers for


    """

    def __init__(self, max_dither=0.7, seed=42, per_night=True, nnights=7305):
        self.survey_features = {}

        self.current_night = -1
        self.max_dither = np.radians(max_dither)
        self.per_night = per_night
        self.rng = np.random.default_rng(seed)
        self.angles = self.rng.random(nnights) * 2 * np.pi
        self.radii = self.max_dither * np.sqrt(self.rng.random(nnights))
        self.offsets = (self.rng.random((nnights, 2)) - 0.5) * 2.0 * self.max_dither
        self.offset = None

    def _generate_offsets(self, n_offsets, night):
        if self.per_night:
            if night != self.current_night:
                self.current_night = night
                self.offset = self.offsets[night, :]
                angle = self.angles[night]
                radius = self.radii[night]
                self.offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            offsets = np.tile(self.offset, (n_offsets, 1))
        else:
            angle = self.rng.random(n_offsets) * 2 * np.pi
            radius = self.max_dither * np.sqrt(self.rng.random(n_offsets))
            offsets = np.array([radius * np.cos(angle), radius * np.sin(angle)]).T

        return offsets

    def __call__(self, observation_list, conditions):
        if len(observation_list) == 0:
            return observation_list
        # Generate offsets in RA and Dec
        offsets = self._generate_offsets(len(observation_list), conditions.night)

        obs_array = np.concatenate(observation_list)
        new_ra, new_dec = gnomonic_project_tosky(
            offsets[:, 0], offsets[:, 1], obs_array["RA"], obs_array["dec"]
        )
        new_ra, new_dec = wrap_ra_dec(new_ra, new_dec)
        for i, obs in enumerate(observation_list):
            observation_list[i]["RA"] = new_ra[i]
            observation_list[i]["dec"] = new_dec[i]
        return observation_list


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
    nnights : `int`
        Number of nights to generate dither positions for.
        Default 7305 (20 years).
    """

    def __init__(
        self,
        dither_bearing_dir=[-0.25, 1],
        dither_bearing_perp=[-0.25, 0.25],
        seed=42,
        per_night=True,
        ra_a=None,
        dec_a=None,
        ra_b=None,
        dec_b=None,
        nnights=7305,
    ):
        self.survey_features = {}

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

        self.current_night = -1

        self.per_night = per_night
        self.shifted_ra_a = None
        self.shifted_dec_a = None
        self.shifted_ra_b = None
        self.shifted_dec_b = None

        rng = np.random.default_rng(seed)
        self.bearings_mag_1 = rng.uniform(
            low=self.dither_bearing_dir.min(),
            high=self.dither_bearing_dir.max(),
            size=nnights,
        )
        self.perp_mag_1 = rng.uniform(
            low=self.dither_bearing_perp.min(),
            high=self.dither_bearing_perp.max(),
            size=nnights,
        )

        self.bearings_mag_2 = rng.uniform(
            low=self.dither_bearing_dir.min(),
            high=self.dither_bearing_dir.max(),
            size=nnights,
        )
        self.perp_mag_2 = rng.uniform(
            low=self.dither_bearing_perp.min(),
            high=self.dither_bearing_perp.max(),
            size=nnights,
        )

    def _generate_offsets(self, n_offsets, night):
        if self.per_night:
            if night != self.current_night:
                self.current_night = night
                bearing_mag = self.bearings_mag_1[night]
                perp_mag = self.perp_mag_1[night]
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
                bearing_mag = self.bearings_mag_2[night]
                perp_mag = self.perp_mag_2[night]

                self.shifted_dec_b, self.shifted_ra_b = dest_latlon(
                    bearing_mag, self.bearing_btoa, self.dec_b, self.ra_b
                )
                self.shifted_dec_b, self.shifted_ra_b = dest_latlon(
                    perp_mag,
                    self.bearing_btoa + np.pi / 2.0,
                    self.shifted_dec_b,
                    self.shifted_ra_b,
                )
        else:
            ValueError("not implamented")

        return (
            self.shifted_ra_a,
            self.shifted_dec_a,
            self.shifted_ra_b,
            self.shifted_dec_b,
        )

    def __call__(self, observation_list, conditions):
        # Generate offsets in RA and Dec
        ra_a, dec_a, ra_b, dec_b = self._generate_offsets(len(observation_list), conditions.night)

        for i, obs in enumerate(observation_list):
            if obs[0]["scheduler_note"][-1] == "a":
                observation_list[i]["RA"] = ra_a
                observation_list[i]["dec"] = dec_a
            elif obs[0]["scheduler_note"][-1] == "b":
                observation_list[i]["RA"] = ra_b
                observation_list[i]["dec"] = dec_b
            else:
                ValueError("observation note does not end in a or b.")
        return observation_list


class CameraRotDetailer(BaseDetailer):
    """
    Randomly set the camera rotation, either for each exposure, or per night.

    Parameters
    ----------
    max_rot : `float` (90.)
        The maximum amount to offset the camera (degrees)
    min_rot : `float` (90)
        The minimum to offset the camera (degrees)
    per_night : `bool` (True)
        If True, only set a new offset per night. If False, randomly
        rotates every observation.
    telescope : `str`
        Telescope name. Options of "rubin" or "auxtel". Default "rubin".
    """

    def __init__(self, max_rot=90.0, min_rot=-90.0, per_night=True, seed=42, nnights=7305, telescope="rubin"):
        self.survey_features = {}

        self.current_night = -1
        self.max_rot = np.radians(max_rot)
        self.min_rot = np.radians(min_rot)
        self.range = self.max_rot - self.min_rot
        self.per_night = per_night
        self.rng = np.random.default_rng(seed)
        self.offsets = self.rng.random(nnights)

        self.offset = None
        self.rc = rotation_converter(telescope=telescope)

    def _generate_offsets(self, n_offsets, night):
        if self.per_night:
            if night != self.current_night:
                self.current_night = night
                self.offset = self.offsets[night] * self.range + self.min_rot
            offsets = np.ones(n_offsets) * self.offset
        else:
            self.rng = np.random.default_rng()
            offsets = self.rng.random(n_offsets) * self.range + self.min_rot

        return offsets

    def __call__(self, observation_list, conditions):
        # Generate offsets in camamera rotator
        offsets = self._generate_offsets(len(observation_list), conditions.night)

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


class CameraSmallRotPerObservationListDetailer(BaseDetailer):
    """
    Randomly set the camera rotation for each observation list.

    Generates a small sequential offset for sequential visits
    in the same filter; adds a random offset for each filter change.

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

    def _generate_offsets_filter_change(self, filter_list, mjd, initial_offset):
        """Generate a random camera rotation for each filter change
        or add a small offset for each sequential observation.
        """
        mjd_hash = round(100 * (np.asarray(mjd).item() % 100))
        rng = np.random.default_rng(mjd_hash * self.seed)

        offsets = np.zeros(len(filter_list))

        # Find the locations of the filter changes
        filter_changes = np.where(np.array(filter_list[:-1]) != np.array(filter_list[1:]))[0]
        filter_changes = np.concatenate([np.array([-1]), filter_changes])
        # But add one because of counting and offsets above.
        filter_changes += 1
        # Count visits per filter in the sequence.
        nvis_per_filter = np.concatenate(
            [np.diff(filter_changes), np.array([len(filter_list) - 1 - filter_changes[-1]])]
        )
        # Set up the random rotator offsets for each filter change
        # This includes first rotation .. maybe not needed?
        for fchange_idx, nvis_f in zip(filter_changes, nvis_per_filter):
            rot_range = self.rot_range - self.per_visit_rot * nvis_f
            # At the filter change spot, update to random offset
            offsets[fchange_idx] = rng.random() * rot_range + self.min_rot
            # After the filter change point, add incremental rotation
            # (we'll wipe this when we get to next fchange_idx)
            offsets[fchange_idx:] += self.per_visit_rot * np.arange(len(filter_list) - fchange_idx)

        return offsets

    def __call__(self, observation_list, conditions):
        # Generate offsets in camera rotator
        filter_list = [np.asarray(obs["filter"]).item() for obs in observation_list]
        offsets = self._generate_offsets_filter_change(filter_list, conditions.mjd, conditions.rot_tel_pos)

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
