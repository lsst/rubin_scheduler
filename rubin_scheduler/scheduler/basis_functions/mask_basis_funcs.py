__all__ = (
    "SolarElongMaskBasisFunction",
    "HaMaskBasisFunction",
    "MoonAvoidanceBasisFunction",
    "MapCloudBasisFunction",
    "PlanetMaskBasisFunction",
    "SolarElongationMaskBasisFunction",
    "AreaCheckMaskBasisFunction",
    "AltAzShadowMaskBasisFunction",
    "AltAzShadowTimeLimitedBasisFunction",
    "RevHaMaskBasisFunction",
    "MaskAllButNES",
    "NInNightMaskBasisFunction",
    "MaskAfterNObsBasisFunction",
    "MaskPoorSeeing",
    "MaskAfterNObsSeeingBasisFunction",
)

import warnings

import healpy as hp
import numpy as np

import rubin_scheduler.scheduler.features as features
from rubin_scheduler.scheduler.basis_functions import BaseBasisFunction
from rubin_scheduler.scheduler.utils import CurrentAreaMap, HpInLsstFov, IntRounded
from rubin_scheduler.utils import DEFAULT_NSIDE, SURVEY_START_MJD, _angular_separation, _hp_grow_mask


class MaskPoorSeeing(BaseBasisFunction):
    """Mask any part of the sky that is below a seeing limit

    Potential to update to take a declination-dependent seeing limit

    Parameters
    ----------
    bandname : `str`
        The band in which to use the seeing value
    seeing_fwhm_max : `float`
        The seeing FWHM effective to use as a limit (arcsec).
    """

    def __init__(self, bandname, seeing_fwhm_max=1.3, nside=DEFAULT_NSIDE):
        super().__init__(nside=nside)
        self.seeing_fwhm_max = seeing_fwhm_max
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.bandname = bandname

    def __call__(self, conditions, indx=None):
        result = self.result.copy()
        to_mask = np.where(conditions.fwhm_eff[self.bandname] > self.seeing_fwhm_max)
        result[to_mask] = np.nan
        return result


class MaskAfterNObsSeeingBasisFunction(BaseBasisFunction):
    """Mask after a HEALpix has been observed N times. In a
    season.

    Parameters
    ----------
    n_max : `int`
        The maximum number of times to obseve in a season. Default 3.
    bandname : `str`
        The bandname. Default None uses all bands.
    seeing_fwhm_max : `float`
        The seeing limit to use when counting observations (arcsec).
    """

    def __init__(
        self,
        n_max=3,
        nside=DEFAULT_NSIDE,
        bandname=None,
        seeing_fwhm_max=1.3,
        mjd_start=SURVEY_START_MJD,
    ):
        super().__init__(nside=nside)
        self.n_max = n_max
        self.survey_features["nobs"] = features.NObservationsCurrentSeason(
            nside=nside,
            bandname=bandname,
            mjd_start=mjd_start,
            seeing_fwhm_max=seeing_fwhm_max,
        )
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        to_mask = np.where(self.survey_features["nobs"].feature >= self.n_max)[0]
        result[to_mask] = np.nan
        return result


class MaskAllButNES(BaseBasisFunction):
    """Mask everything but the NES region

    Parameters
    ----------
    nes_region : `np.array`
        The index values for HEALpix maps that
        correspond to the NES region. Default None
        uses current default map to label NES.
    """

    def __init__(self, nes_region=None, nside=DEFAULT_NSIDE):
        super().__init__(nside=nside)

        if nes_region is None:
            sag = CurrentAreaMap()
            sky_maps, labels = sag.return_maps()

            nes_region = np.where(labels != "nes")[0]

        self.indx = nes_region
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)

    def _calc_value(self, conditions, **kwargs):
        result = self.result.copy()
        return result


class NInNightMaskBasisFunction(BaseBasisFunction):
    """Mask a HEALpix for the rest of the night
    after it has been observed N times.

    Parameters
    ----------
    n_limit : `int`
        Number of visits to mask after. Default 1.
    bandname : `str`
        Which band to use. Default None assumes all bands.
    """

    def __init__(self, nside=DEFAULT_NSIDE, n_limit=1, bandname=None):

        super(NInNightMaskBasisFunction, self).__init__(nside=nside)
        self.n_limit = n_limit
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.survey_features["n_in_night"] = features.NObsNight(nside=nside, bandname=bandname)

    def _calc_value(self, conditions, indx=None):

        result = self.result.copy()
        to_mask = np.where(self.survey_features["n_in_night"].feature >= self.n_limit)[0]
        result[to_mask] = np.nan
        return result


class RevHaMaskBasisFunction(BaseBasisFunction):
    """Limit the sky based on hour angle

    Parameters
    ----------
    ha_min : float (None)
        The minimum hour angle to accept (hours)
    ha_max : float (None)
        The maximum hour angle to accept (hours)
    """

    def __init__(self, ha_min=None, ha_max=None, nside=DEFAULT_NSIDE):
        super(RevHaMaskBasisFunction, self).__init__(nside=nside)
        self.ha_max = ha_max
        self.ha_min = ha_min
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float) + np.nan

    def _calc_value(self, conditions, **kwargs):
        result = self.result.copy()

        if self.ha_min is not None:
            good = np.where(conditions.HA < (self.ha_min / 12.0 * np.pi))[0]
            result[good] = 0
        if self.ha_max is not None:
            good = np.where(conditions.HA > (self.ha_max / 12.0 * np.pi))[0]
            result[good] = 0

        return result


class MaskAfterNObsBasisFunction(BaseBasisFunction):
    """Mask after a HEALpix has been observed N times.

    Parameters
    ----------
    n_max : `int`
        The maximum number of times. Default 3.
    bandname : `str`
        The bandname. Default None uses all bands.
    """

    def __init__(self, n_max=3, nside=DEFAULT_NSIDE, bandname=None):
        super(MaskAfterNObsBasisFunction, self).__init__(nside=nside)
        self.n_max = n_max
        self.survey_features["nobs"] = features.NObservations(nside=nside, bandname=bandname)
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        to_mask = np.where(self.survey_features["nobs"].feature >= self.n_max)[0]
        result[to_mask] = np.nan
        return result


class SolarElongMaskBasisFunction(BaseBasisFunction):
    """Mask regions larger than some solar elongation limit

    Parameters
    ----------
    elong_limit : float (45)
        The limit beyond which to mask (degrees)
    """

    def __init__(self, elong_limit=45.0, nside=DEFAULT_NSIDE):
        msg = (
            "SolarElongMaskBasisFunction is scheduled for"
            " deprecation, swap to SolarElongationMaskBasisFunction."
        )
        warnings.warn(msg, DeprecationWarning)
        super(SolarElongMaskBasisFunction, self).__init__(nside=nside)
        self.elong_limit = IntRounded(np.radians(elong_limit))
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        to_mask = np.where(IntRounded(conditions.solar_elongation) > self.elong_limit)[0]
        result[to_mask] = np.nan
        return result


class HaMaskBasisFunction(BaseBasisFunction):
    """Limit the sky based on hour angle

    Parameters
    ----------
    ha_min : float (None)
        The minimum hour angle to accept (hours)
    ha_max : float (None)
        The maximum hour angle to accept (hours)
    """

    def __init__(self, ha_min=None, ha_max=None, nside=DEFAULT_NSIDE):
        super(HaMaskBasisFunction, self).__init__(nside=nside)
        self.ha_max = ha_max
        self.ha_min = ha_min
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)

    def _calc_value(self, conditions, **kwargs):
        result = self.result.copy()

        if self.ha_min is not None:
            good = np.where(conditions.HA < (self.ha_min / 12.0 * np.pi))[0]
            result[good] = np.nan
        if self.ha_max is not None:
            good = np.where(conditions.HA > (self.ha_max / 12.0 * np.pi))[0]
            result[good] = np.nan

        return result


class AreaCheckMaskBasisFunction(BaseBasisFunction):
    """Take a list of other mask basis functions, and do an additional
    check for area available"""

    def __init__(self, bf_list, nside=DEFAULT_NSIDE, min_area=1000.0):
        super(AreaCheckMaskBasisFunction, self).__init__(nside=nside)
        self.bf_list = bf_list
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.min_area = min_area

    def check_feasibility(self, conditions):
        result = True
        for bf in self.bf_list:
            if not bf.check_feasibility(conditions):
                return False

        area_map = self.result.copy()
        for bf in self.bf_list:
            area_map *= bf(conditions)

        good_pix = np.where(area_map == 0)[0]
        if hp.nside2pixarea(self.nside, degrees=True) * good_pix.size < self.min_area:
            result = False
        return result

    def _calc_value(self, conditions, **kwargs):
        result = self.result.copy()
        for bf in self.bf_list:
            result *= bf(conditions)
        return result


class SolarElongationMaskBasisFunction(BaseBasisFunction):
    """Mask things at various solar elongations

    Parameters
    ----------
    min_elong : float (0)
        The minimum solar elongation to consider (degrees).
    max_elong : float (60.)
        The maximum solar elongation to consider (degrees).
    """

    def __init__(self, min_elong=0.0, max_elong=60.0, nside=DEFAULT_NSIDE, penalty=np.nan):
        super(SolarElongationMaskBasisFunction, self).__init__(nside=nside)
        self.min_elong = np.radians(min_elong)
        self.max_elong = np.radians(max_elong)
        self.penalty = penalty
        self.result = np.empty(hp.nside2npix(self.nside), dtype=float)
        self.result.fill(self.penalty)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        in_range = np.where(
            (IntRounded(conditions.solar_elongation) >= IntRounded(self.min_elong))
            & (IntRounded(conditions.solar_elongation) <= IntRounded(self.max_elong))
        )[0]
        result[in_range] = 1
        return result


class PlanetMaskBasisFunction(BaseBasisFunction):
    """Mask the bright planets.

    Parameters
    ----------
    mask_radius : float (3.5)
        The radius to mask around a planet (degrees).
    planets : list of str (None)
        A list of planet names to mask. Defaults to ['venus', 'mars',
        'jupiter']. Not including Saturn because it moves really slowly
        and has average apparent mag of ~0.4, so fainter than Vega.

    """

    def __init__(self, mask_radius=3.5, planets=None, nside=DEFAULT_NSIDE, scale=1e5):
        super(PlanetMaskBasisFunction, self).__init__(nside=nside)
        if planets is None:
            planets = ["venus", "mars", "jupiter"]
        self.planets = planets
        self.mask_radius = np.radians(mask_radius)
        self.result = np.zeros(hp.nside2npix(nside))
        # set up a kdtree. Could maybe use healpy.query_disc instead.
        self.in_fov = HpInLsstFov(nside=nside, fov_radius=mask_radius, scale=scale)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        for pn in self.planets:
            indices = self.in_fov(
                np.max(conditions.planet_positions[pn + "_RA"]),
                np.max(conditions.planet_positions[pn + "_dec"]),
            )
            result[indices] = np.nan

        return result


class AltAzShadowMaskBasisFunction(BaseBasisFunction):
    """Mask out a range of altitudes and azimuths, including
     regions which will enter the mask within `shadow_minutes`.

    Masks any alt/az regions as specified by the conditions object in
    `conditions.sky_az_limits` and `conditions.sky_alt_limits`,
    as well as values as provided by `min_alt`, `max_alt`,
    `min_az` and `max_az`.

    Parameters
    ----------
    nside : `int` or None
        HEALpix nside. Default None will look up the package-wide default.
    min_alt : `float`
        Minimum altitude to apply to the mask. Default 20 (degrees).
    max_alt : `float`
        Maximum altitude to allow. Default 82 (degrees).
    min_az : `float`
        Minimum azimuth value to apply to the mask. Default 0 (degrees).
        These azimuth values are absolute azimuth, not cumulative.
        The absolute and cumulative azimuth only diverge if the azimuth
        range is greater than 360 degrees.
    max_az : `float`
        Maximum azimuth value to apply to the mask. Default 360 (degrees).
    shadow_minutes : `float`
        How long to extend masked area in longitude. Default 40 (minutes).
        Choose this value based on the time between when a field might
        be chosen to be scheduled and when it might be observed.
        For pairs, the minimum pair time + some buffer is good.
        For sequences, try the length of the sequence + some buffer.
    pad : `float`
        The value by which to pad the telescope limits, to avoid
        healpix values mapping into pointings from the field tesselations
        which are actually out of bounds. This should typically be
        a bit more than the radius of the fov.  (degrees).
    time_step : `float`
        Calculate mask at multiple timesteps. Useful for when
        shadow minutes is large. Default 10 (minutes).
    """

    def __init__(
        self,
        nside=DEFAULT_NSIDE,
        min_alt=20.0,
        max_alt=86.5,
        min_az=0,
        max_az=360,
        shadow_minutes=40.0,
        pad=3.0,
        scale=1000,
        time_step=10.0,
    ):
        super().__init__(nside=nside)
        self.min_alt = np.radians(min_alt)
        self.max_alt = np.radians(max_alt)
        self.min_az = np.radians(min_az)
        self.max_az = np.radians(max_az)
        self.shadow_time = shadow_minutes / 60.0 / 24.0  # To days
        self.time_step = time_step / 60.0 / 24.0
        self.pad = np.radians(pad)

        self.r_min_alt = IntRounded(self.min_alt)
        self.r_max_alt = IntRounded(self.max_alt)
        self.scale = scale

    def _calc_value(self, conditions, indx=None):
        # Basis function value will be 0 except where masked (then np.nan)
        result = np.zeros(hp.nside2npix(self.nside), dtype=float)

        # For backwards compatiibility:
        try:
            n_steps = np.floor(self.shadow_time / self.time_step)
        except AttributeError:
            n_steps = 0

        if n_steps > 0:
            times = np.arange(n_steps) * self.time_step
            times = times.tolist()
            if self.shadow_time not in times:
                times.append(self.shadow_time)
        else:
            times = [self.shadow_time]

        for time in times:
            # Compute the alt,az values in the future. Use the conditions
            # object so the results are cached and can be used by other
            # surveys is needed. Technically this could fail if the masked
            # region is very narrow or shadow time is very large.
            future_alt, future_az = conditions.future_alt_az(float(np.max(conditions.mjd)) + time)
            r_future_alt = IntRounded(future_alt)
            r_current_alt = IntRounded(conditions.alt)

            # Check the basis function altitude limits, now and future
            result[np.where(r_current_alt < self.r_min_alt)] = np.nan
            result[np.where(r_current_alt > self.r_max_alt)] = np.nan
            result[np.where(r_future_alt < self.r_min_alt)] = np.nan
            result[np.where(r_future_alt > self.r_max_alt)] = np.nan

            # Check the conditions objects 'sky_alt_limit', now and future
            if (conditions.sky_alt_limits is not None) and (len(conditions.sky_alt_limits) > 0):
                combined = np.zeros(hp.nside2npix(self.nside), dtype=float)
                for limits in conditions.sky_alt_limits:
                    # For conditions-based limits, must add pad
                    # And remember that discontinuous areas can be allowed
                    in_bounds = np.ones(hp.nside2npix(self.nside), dtype=float)
                    min_alt = IntRounded(limits[0])
                    max_alt = IntRounded(limits[1])
                    in_bounds[np.where(r_current_alt < min_alt)] = 0
                    in_bounds[np.where(r_current_alt > max_alt)] = 0
                    in_bounds[np.where(r_future_alt < min_alt)] = 0
                    in_bounds[np.where(r_future_alt > max_alt)] = 0
                    combined += in_bounds
                result[np.where(combined == 0)] = np.nan
            # And check against the telescope 'tel_alt_limits'.
            # The tel_alt_limits could be combined with the sky_alt_limits,
            # but it's a bit tricky because sky_alt_limits are (potentially)
            # disjoint allowable areas, while the tel_alt_limits are a single
            # wide set of allowable area which explicitly disallows anything
            # outside that range. The az limits versions are similar.
            if conditions.tel_alt_limits is not None:
                min_alt = IntRounded(conditions.tel_alt_limits[0])
                max_alt = IntRounded(conditions.tel_alt_limits[1])
                result[np.where(r_current_alt < min_alt)] = np.nan
                result[np.where(r_current_alt > max_alt)] = np.nan
                result[np.where(r_future_alt < min_alt)] = np.nan
                result[np.where(r_future_alt > max_alt)] = np.nan

            # note that % (mod) is not supported for IntRounded
            two_pi = 2 * np.pi
            # Check the basis function azimuth limits, now and future
            if np.abs(self.max_az - self.min_az) < two_pi:
                az_range = (self.max_az - self.min_az) % (two_pi)
                out_of_bounds = np.where((conditions.az - self.min_az) % (two_pi) > az_range)[0]
                result[out_of_bounds] = np.nan
                out_of_bounds = np.where((future_az - self.min_az) % (two_pi) > az_range)[0]
                result[out_of_bounds] = np.nan
            # Check the conditions objects azimuth limits, now and future
            if (conditions.sky_az_limits is not None) and (len(conditions.sky_az_limits) > 0):
                combined = np.zeros(hp.nside2npix(self.nside), dtype=float)
                for limits in conditions.sky_az_limits:
                    in_bounds = np.ones(hp.nside2npix(self.nside), dtype=float)
                    min_az = limits[0]
                    max_az = limits[1]
                    if np.abs(max_az - min_az) < two_pi:
                        az_range = (max_az - min_az) % (two_pi)
                        out_of_bounds = np.where((conditions.az - min_az) % (two_pi) > az_range)[0]
                        in_bounds[out_of_bounds] = 0
                        out_of_bounds = np.where((future_az - min_az) % (two_pi) > az_range)[0]
                        in_bounds[out_of_bounds] = 0
                    combined += in_bounds
                result[np.where(combined == 0)] = np.nan
            # Check against the kinematic hard limits.
            if conditions.tel_az_limits is not None:
                if np.abs(conditions.tel_az_limits[1] - conditions.tel_az_limits[0]) < two_pi:
                    az_range = (conditions.tel_az_limits[1] - conditions.tel_az_limits[0]) % (two_pi)
                    out_of_bounds = np.where(
                        (conditions.az - conditions.tel_az_limits[0]) % (two_pi) > az_range
                    )[0]
                    result[out_of_bounds] = np.nan
                    out_of_bounds = np.where((future_az - conditions.tel_az_limits[0]) % (two_pi) > az_range)[
                        0
                    ]
                    result[out_of_bounds] = np.nan

        # Grow the resulting mask by self.pad, to avoid field centers
        # falling outside the boundaries of what is actually reachable.
        if self.pad > 0:
            mask_indx = np.where(np.isnan(result))[0]
            to_mask_indx = _hp_grow_mask(self.nside, tuple(mask_indx), scale=self.scale, grow_size=self.pad)
            result[to_mask_indx] = np.nan

        return result


class AltAzShadowTimeLimitedBasisFunction(AltAzShadowMaskBasisFunction):
    """Similar to AltAzShadowMaskBasisFunction, but only adds a mask
    if the time within the night is close to the specified sunrise/sunset
    event

    Parameters
    ----------
    <all the kwargs from AltAzShadowMaskBasisFunction>

    time_to_sun : `float`
        How close in time one must be to the sun up/down time (hours).
        Default 2.
    sun_keys : `list`
        List of strings that are looked up in the conditions object.
        Default ["sunrise", "sunset"]. Options currently are
        sunrise, sunset, sun_n12_setting, sun_n18_setting,
        sun_n18_rising, sun_n12_rising
    """

    def __init__(self, time_to_sun=2.0, sun_keys=["sunrise", "sunset"], **kwargs):
        self.time_to_sun = time_to_sun / 24.0  # to days
        self.sun_keys = sun_keys
        super().__init__(**kwargs)

    def _calc_value(self, conditions, indx=None):

        time_to_sun = [conditions.mjd - getattr(conditions, key) for key in self.sun_keys]
        time_to_sun = np.abs(time_to_sun)
        # We are close to a sunrise/set event, apply mask
        if np.min(time_to_sun) < self.time_to_sun:
            return super()._calc_value(conditions, indx=indx)
        else:
            # In the middle of the night, no mask
            return 0


class MoonAvoidanceBasisFunction(BaseBasisFunction):
    """Avoid observing within `moon_distance` of the moon.

    Parameters
    ----------
    moon_distance: float (30.)
        Minimum allowed moon distance. (degrees)

    Notes
    -----
    As the current specified requirements for the observatory
    are "observe more than 30 degrees from the moon", this basis
    function simply does that at present. This includes
    times the moon is below the horizon or if the moon close to new.
    Most likely, this avoidance region should depend on lunar phase,
    the band used for observations, and whether the moon is above
    or below the horizon.
    """

    def __init__(self, nside=DEFAULT_NSIDE, moon_distance=30.0):
        super(MoonAvoidanceBasisFunction, self).__init__(nside=nside)
        self.update_on_newobs = False

        self.moon_distance = IntRounded(np.radians(moon_distance))
        self.result = np.ones(hp.nside2npix(self.nside), dtype=float)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()

        angular_distance = _angular_separation(
            conditions.az, conditions.alt, conditions.moon_az, conditions.moon_alt
        )

        result[IntRounded(angular_distance) < self.moon_distance] = np.nan

        return result


class BulkCloudBasisFunction(BaseBasisFunction):
    """Mark healpixels on a map if their cloud values are greater than
    the same healpixels on a maximum cloud map.

    Parameters
    ----------
    nside: int (default_nside)
        The healpix resolution.
    max_cloud_map : numpy array (None)
        A healpix map showing the maximum allowed cloud values for all
        points on the sky
    out_of_bounds_val : float (10.)
        Point value to give regions where there are no observations
        requested
    """

    def __init__(self, nside=DEFAULT_NSIDE, max_cloud_map=None, max_val=0.7, out_of_bounds_val=np.nan):
        super(BulkCloudBasisFunction, self).__init__(nside=nside)
        self.update_on_newobs = False

        if max_cloud_map is None:
            self.max_cloud_map = np.zeros(hp.nside2npix(nside), dtype=float) + max_val
        else:
            self.max_cloud_map = max_cloud_map
        self.out_of_bounds_area = np.where(self.max_cloud_map > 1.0)[0]
        self.out_of_bounds_val = out_of_bounds_val
        self.result = np.ones(hp.nside2npix(self.nside))

    def _calc_value(self, conditions, indx=None):
        """
        Parameters
        ----------
        indx : list (None)
            Index values to compute, if None, full map is computed
        Returns
        -------
        Healpix map where pixels with a cloud value greater than the
        max_cloud_map value are marked as unseen.
        """

        result = self.result.copy()

        clouded = np.where(self.max_cloud_map <= conditions.bulk_cloud)
        result[clouded] = self.out_of_bounds_val

        return result


class MapCloudBasisFunction(BaseBasisFunction):
    """Mark healpixels on a map if their cloud values are greater than
    the same healpixels on a maximum cloud map. Currently a placeholder for
    when the telemetry stream can include a full sky cloud map.

    Parameters
    ----------
    nside: int (default_nside)
        The healpix resolution.
    max_cloud_map : numpy array (None)
        A healpix map showing the maximum allowed cloud values for all
        points on the sky
    out_of_bounds_val : float (10.)
        Point value to give regions where there are no observations
        requested
    """

    def __init__(self, nside=DEFAULT_NSIDE, max_cloud_map=None, max_val=0.7, out_of_bounds_val=np.nan):
        super().__init__(nside=nside)
        self.update_on_newobs = False

        if max_cloud_map is None:
            self.max_cloud_map = np.zeros(hp.nside2npix(nside), dtype=float) + max_val
        else:
            self.max_cloud_map = max_cloud_map
        self.out_of_bounds_area = np.where(self.max_cloud_map > 1.0)[0]
        self.out_of_bounds_val = out_of_bounds_val
        self.result = np.ones(hp.nside2npix(self.nside))

    def _calc_value(self, conditions, indx=None):
        """
        Parameters
        ----------
        indx : list (None)
            Index values to compute, if None, full map is computed
        Returns
        -------
        Healpix map where pixels with a cloud value greater than the
        max_cloud_map value are marked as unseen.
        """

        result = self.result.copy()

        clouded = np.where(self.max_cloud_map <= conditions.bulk_cloud)
        result[clouded] = self.out_of_bounds_val

        return result
