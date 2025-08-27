__all__ = ("Conditions",)

import functools
import warnings
from inspect import signature
from io import StringIO

import healpy as hp
import numpy as np
import pandas as pd

from rubin_scheduler.scheduler.utils import match_hp_resolution, smallest_signed_angle
from rubin_scheduler.utils import (
    DEFAULT_NSIDE,
    SURVEY_START_MJD,
    Site,
    _angular_separation,
    _approx_altaz2pa,
    _approx_ra_dec2_alt_az,
    _hpid2_ra_dec,
    calc_lmst,
    m5_flat_sed,
)


class Conditions:
    """Holds telemetry information, keeping calculated values in sync
    for `self.mjd` (such as ra/dec mappings to alt/az).

    Incoming values have setters to keep values in sync. Healpix maps
    are set to the expected (`self.nside`) resolution.

    Unless otherwise noted, all values are assumed to be valid at the time
    given by `self.mjd`.

    Parameters
    ----------
    nside : `int`, optional
        The healpixel nside to set the resolution of attributes.
        Default of None will use
        `rubin_scheduler.scheduler.utils.set_default_nside`.
    site : str ('LSST')
        A site name used to create a sims.utils.Site object. For
        looking up observatory parameters like latitude and longitude.
    exptime : `float`, optional
        The exposure time (seconds) to assume when computing the 5-sigma
        limiting depth maps stored in Conditions. These maps are used
        by basis functions, but are not used to calculate expected
        observation m5 values (such as when exposure time varies).
        Default 30 seconds.
    mjd : `float`, optional
        The current MJD.
        Default of None is fine on init - will be updated by telemetry
        stream.
    """

    global_maps = set(["ra", "dec", "slewtime", "airmass", "zeros_map", "nan_map"])
    by_band_maps = set(["skybrightness", "fwhm_eff", "m5_depth"])

    def __init__(
        self,
        nside=DEFAULT_NSIDE,
        site="LSST",
        exptime=30.0,
        mjd=None,
        survey_start_mjd=SURVEY_START_MJD,
    ):
        """
        Attributes (Set on init)
        -----------
        nside : `int`
            Healpix resolution. All maps are set to this reslution.
        site : rubin_scheduler.Site object ('LSST')
            Contains static site-specific data (lat, lon, altitude, etc).
            Defaults to 'LSST'.
        ra : `np.ndarray`, (N,)
            A healpix array with the RA of each healpixel center (radians).
            Automatically generated.
        dec : `np.ndarray`, (N,)
            A healpix array with the Dec of each healpixel center (radians).
            Automatically generated.
        survey_start_mjd : `float`
            The starting MJD of the survey. Should be durring the day
            before the first night of observing.

        Attributes (to be set by user/telemetry stream/scheduler)
        -------------------------------------------
        mjd : `float`
            Modified Julian Date (days).
        bulk_cloud : `float`
            The fraction of sky covered by clouds. Generally only
            set in simulations. Not currently being used, so probably
            due for deprecation.
        cloud_maps : `CloudMap`
            rubin_scheduler.site_models.CloudMap object.
        slewtime : `np.ndarray`, (N,)
            Healpix showing the slewtime to each healpixel center (seconds)
        current_band : `str`
            The name of the current band. (expect one of u, g, r, i, z, y).
        current_filter : `str`
            Deprecated version of current_band.
        mounted_bands : `list` [`str`]
            The bands that are currently mounted and thus available
            (expect 5 of u, g, r, i, z, y for LSSTCam).
        mounted_filters : `list` [`str`]
            Deprecated version of mounted_bands.
        skybrightness : `dict` {`str: `np.ndarray`, (N,)}
            Dictionary keyed by band name.
            Values are healpix arrays with the sky brightness at each
            healpix center (mag/acsec^2)
        fwhm_eff : `dict` {`str: `np.ndarray`, (N,)}
            Dictionary keyed by bandname.
            Values are the effective seeing FWHM at each healpix
            center (arcseconds)
        moon_alt : `float`
            The altitude of the Moon (radians)
        moon_az : `float`
            The Azimuth of the moon (radians)
        moon_ra : `float`
            RA of the moon (radians)
        moon_dec : `float`
            Declination of the moon (radians)
        moon_phase : `float`
            The Phase of the moon. (percent, 0=new moon, 100=full moon)
        sun_alt : `float`
            The altitude of the sun (radians).
        sun_az : `float`
            The Azimuth of the sun (radians).
        sun_ra : `float`
            The RA of the sun (radians).
        sun_dec : `float`
            The Dec of the sun (radians).
        tel_ra : `float`
            The current telescope RA pointing (radians).
        tel_dec : `float`
            The current telescope Declination (radians).
        tel_alt : `float`
            The current telescope altitude (radians).
        tel_az : `float`
            The current telescope azimuth (radians).
        cumulative_azimuth_rad : `float`
            The cumulative telescope azimuth (radians).
            Used for tracking cable wrap.
        wind_speed : `float`
            Wind speed (m/s).
        wind_direction : `float`
            Direction from which the wind originates. A direction of
            0.0 degrees means the wind originates from the north and 90.0
            degrees from the east (radians).
        sunset : `float`
            The MJD of sunset that starts the current night. Note MJDs of
            sunset, moonset, twilight times, etc are from interpolations.
            This means the sun may actually be slightly above/below the
            horizon at the given sunset time.
        sun_n12_setting : `float`
            The MJD of when the sun is at -12 degrees altitude and
            setting during the current night. From interpolation.
        sun_n18_setting : `float`
            The MJD when the sun is at -18 degrees altitude and setting
            during the current night. From interpolation.
        sun_n18_rising : `float`
            The MJD when the sun is at -18 degrees altitude and rising
            during the current night. From interpolation.
        sun_n12_rising : `float`
            The MJD when the sun is at -12 degrees altitude and rising
            during the current night. From interpolation.
        sunrise : `float`
            The MJD of sunrise during the current night. From interpolation
        moonrise : `float`
            The MJD of moonrise during the current night. From interpolation.
        moonset : `float`
            The MJD of moon set during the current night. From interpolation.
        moon_phase_sunset : `float`
            The phase of the moon (0-100 illuminated) at sunset.
            Useful for setting which bands should be loaded.
        targets_of_opportunity : `list` [`rubin_scheduler.scheduler.targetoO`]
            targetoO objects.
        planet_positions : `dict` {`str`: `float`}
            Dictionary of planet name and coordinate e.g., 'venus_RA',
            'mars_dec'
        scheduled_observations : `np.ndarray`, (M,)
            A list of MJD times when there are scheduled observations.
            Defaults to empty array.
        sky_az_limits : `list` [[`float`, `float`]]
            A list of lists giving valid azimuth ranges. e.g.,
            [0, 2*np.pi] would mean all azimuth values are valid, while
            [[0, np.pi/2], [3*np.pi/2, 2*np.pi]] would mean anywhere in
            the south is invalid.  Radians.
        sky_alt_limits : `list` [[`float`, `float`]]
            A list of lists giving valid altitude ranges. Radians.
        tel_az_limits : `list` [`float`, `float`]
            A simple two-element list giving the valid azimuth ranges
            for the telescope movement. Radians.
        tel_alt_limits : `list` [`float`, `float`]
            A simple two-element list giving the valid altitude ranges
            for the telescope movement. Radians

        Attributes (calculated on demand and cached)
        ------------------------------------------
        alt : `np.ndarray`, (N,)
            Altitude of each healpixel (radians). Recalculated if mjd is
            updated. Uses fast approximate equation for converting
            RA,Dec to alt,az.
        az : `np.ndarray`, (N,)
            Azimuth of each healpixel (radians). Recalculated if mjd is
            updated. Uses fast approximate equation for converting RA,Dec
            to alt,az.
        airmass : `np.ndarray`, (N,)
            A healpix map with the airmass value of each healpixel. (unitless)
        pa : `np.ndarray`, (N,)
            The parallactic angle of each healpixel (radians). Recalculated
            if mjd is updated. Based on the fast approximate alt,az values.
        lmst : `float`
            The local mean sidereal time (hours). Updates is mjd is changed.
        m5_depth : `dict` {`str: `np.ndarray`, (N,)}
            the 5-sigma limiting depth healpix maps, keyed by bandname
            (mags). Will be recalculated if the skybrightness, seeing,
            or airmass are updated.
        HA : `np.ndarray`, (N,)
            Healpix map of the hour angle of each healpixel (radians).
            Runs from 0 to 2pi.
        az_to_sun : `np.ndarray`, (N,)
            Healpix map of the azimuthal distance to the sun for each
            healpixel (radians)
        az_to_anitsun : `np.ndarray`, (N,)
            Healpix map of the azimuthal distance to the anit-sun for each
            healpixel (radians)
        solar_elongation : `np.ndarray`, (N,)
            Healpix map of the solar elongation (angular distance to the sun)
            for each healpixel (radians)
        night : `int`
            The current night number (days).

        Attributes (set by the scheduler)
        -------------------------------
        queue : `list` [`observation` objects]
            The current queue of observations core_scheduler is waiting to
            execute.

        """
        self.nside = nside
        self.survey_start_mjd = survey_start_mjd

        # The RA, Dec grid we are using
        hpids = np.arange(hp.nside2npix(self.nside))
        self.ra, self.dec = _hpid2_ra_dec(self.nside, hpids)

        self.site = Site(site)
        self.exptime = exptime

        # Generate an empty map so we can copy when we need a new map
        self.zeros_map = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.nan_map = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.nan_map.fill(np.nan)

        # Set other attributes to Nones
        self._init_attributes()
        self.mjd = mjd

    def _init_attributes(self):
        """Initialize or clear all the attributes"""

        # Modified Julian Date (day)
        self._mjd = None
        # Altitude and azimuth. Dict with degrees and radians
        self._alt = None
        self._az = None
        self._pa = None

        # The cloud level. Fraction, but could upgrade to transparency map
        self.clouds = None
        self._slewtime = None
        self.current_band = None
        self._current_filter = None
        self.mounted_bands = None
        self._mounted_filters = None
        self._lmst = None
        # Should be a dict with bandname keys
        self._skybrightness = {}
        self._fwhm_eff = {}
        self._m5_depth = None
        self._airmass = None

        self.wind_speed = None
        self.wind_direction = None

        # Upcoming scheduled observations
        self.scheduled_observations = np.array([], dtype=float)

        # Attribute to hold the current observing queue
        self.queue = None

        # Moon
        self.moon_alt = None
        self.moon_az = None
        self.moon_ra = None
        self.moon_dec = None
        self.moon_phase = None

        # Sun
        self.sun_alt = None
        self.sun_az = None
        self.sun_ra = None
        self.sun_dec = None

        # Almanac information
        self.sunset = None
        self.sun_n12_setting = None
        self.sun_n18_setting = None
        self.sun_n18_rising = None
        self.sun_n12_rising = None
        self.sunrise = None
        self.moonrise = None
        self.moonset = None
        self.moon_phase_sunset = None

        self.planet_positions = None

        # Current telescope pointing
        self.tel_ra = None
        self.tel_dec = None
        self.tel_alt = None
        self.tel_az = None
        self.cumulative_azimuth_rad = None

        # Sky coverage limits.
        # These should be in radians.
        self.sky_az_limits = None
        self.sky_alt_limits = None
        # Kinematic model (real slew) limits.
        self.tel_alt_limits = None
        self.tel_az_limits = None

        # Full sky cloud map object
        self.cloud_maps = None
        self._HA = None

        # XXX--document
        self.bulk_cloud = None

        self.rot_tel_pos = None

        self.targets_of_opportunity = None

        # Potential attributes that get computed
        self._solar_elongation = None
        self._az_to_sun = None
        self._az_to_antisun = None

    # We can get away with only using current_filter and mounted_filter
    # with getter/setter because we don't have to shim back to these
    # values *from* band, only from filter toward band.
    # Also use of Conditions within FBS should use *band* not filter.
    @property
    def current_filter(self):
        return self._current_filter

    @current_filter.setter
    def current_filter(self, value):
        self._current_filter = value
        self.current_band = value.split("_")[0]

    @property
    def mounted_filters(self):
        return self._mounted_filters

    @mounted_filters.setter
    def mounted_filters(self, value):
        self._mounted_filters = value
        self.mounted_bands = [v.split("_")[0] for v in value]

    @property
    def lmst(self):
        if self._lmst is None:
            self._lmst = calc_lmst(self.mjd, self.site.longitude_rad)

        return self._lmst

    @lmst.setter
    def lmst(self, value):
        self._lmst = value
        self._HA = None

    @property
    def HA(self):
        if self._HA is None:
            self.calc_ha()
        return self._HA

    def calc_ha(self):
        self._HA = np.radians(self.lmst * 360.0 / 24.0) - self.ra
        self._HA[np.where(self._HA < 0)] += 2.0 * np.pi

    @property
    def slewtime(self):
        return self._slewtime

    @slewtime.setter
    def slewtime(self, value):
        # Using 0 for start of night
        if np.size(value) == 1:
            self._slewtime = value
        else:
            self._slewtime = match_hp_resolution(value, nside_out=self.nside)

    @property
    def airmass(self):
        if self._airmass is None:
            self.calc_airmass()
        return self._airmass

    def calc_airmass(self):
        self._airmass = np.zeros(self._alt.size, dtype=float)
        self._airmass.fill(np.nan)
        alt_limit = self.tel_alt_limits
        if alt_limit is None:
            alt_limit = 0
        else:
            alt_limit = np.min(alt_limit)
        good = np.where(self._alt > alt_limit)
        self._airmass[good] = 1.0 / np.cos(np.pi / 2.0 - self._alt[good])

    @property
    def pa(self):
        if self._pa is None:
            self.calc_pa()
        return self._pa

    def calc_pa(self):
        self._pa = _approx_altaz2pa(self.alt, self.az, self.site.latitude_rad)

    @property
    def alt(self):
        if self._alt is None:
            self.calc_alt_az()
        return self._alt

    @property
    def az(self):
        if self._az is None:
            self.calc_alt_az()
        return self._az

    def calc_alt_az(self):
        self._alt, self._az = _approx_ra_dec2_alt_az(
            self.ra,
            self.dec,
            self.site.latitude_rad,
            self.site.longitude_rad,
            self._mjd,
        )

    @functools.lru_cache(maxsize=10)
    def future_alt_az(self, mjd):
        """Compute the altitude and azimuth for a future time.

        Returns
        -------
        altitude : `np.array`
            The altutude of each healpix at MJD (radians)
        azimuth : : `np.array`
            The azimuth of each healpix at MJD (radians)
        """
        alt, az = _approx_ra_dec2_alt_az(
            self.ra,
            self.dec,
            self.site.latitude_rad,
            self.site.longitude_rad,
            mjd,
        )
        return alt, az

    @property
    def mjd(self):
        return self._mjd

    @mjd.setter
    def mjd(self, value):
        # If MJD is changed, everything else is no longer valid, so re-init
        if self.mjd is not None:
            warnings.warn("Changing MJD and resetting all attributes.")
        self._init_attributes()
        self._mjd = value

    @property
    def skybrightness(self):
        return self._skybrightness

    @skybrightness.setter
    def skybrightness(self, indict):
        for key in indict:
            self._skybrightness[key] = match_hp_resolution(indict[key], nside_out=self.nside)
        # If sky brightness changes, need to recalc M5 depth.
        self._m5_depth = None

    @property
    def fwhm_eff(self):
        return self._fwhm_eff

    @fwhm_eff.setter
    def fwhm_eff(self, indict):
        for key in indict:
            self._fwhm_eff[key] = match_hp_resolution(indict[key], nside_out=self.nside)
        self._m5_depth = None

    @property
    def night(self):
        self.calc_night()
        # Take a max to make sure we strip off any strange array structure
        return np.max(self._night)

    def calc_night(self):
        self._night = np.floor(self.mjd - self.survey_start_mjd).astype(int)

    @night.setter
    def night(self, val):
        msg = (
            "Attribute 'night' no longer to be set by users. "
            "Set survey_start_mjd on init and mjd so Conditions"
            "can compute value for 'night'."
        )
        warnings.warn(msg, DeprecationWarning)

    @property
    def m5_depth(self):
        if self._m5_depth is None:
            self.calc_m5_depth()
        return self._m5_depth

    def calc_m5_depth(self):
        self._m5_depth = {}
        for bandname in self._skybrightness:
            good = ~np.isnan(self._skybrightness[bandname])
            self._m5_depth[bandname] = self.nan_map.copy()
            self._m5_depth[bandname][good] = m5_flat_sed(
                bandname,
                self._skybrightness[bandname][good],
                self._fwhm_eff[bandname][good],
                self.exptime,
                self._airmass[good],
            )

    def calc_solar_elongation(self):
        self._solar_elongation = _angular_separation(self.ra, self.dec, self.sun_ra, self.sun_dec)

    @property
    def solar_elongation(self):
        if self._solar_elongation is None:
            self.calc_solar_elongation()
        return self._solar_elongation

    def calc_az_to_sun(self):
        self._az_to_sun = smallest_signed_angle(self.ra, self.sun_ra)

    def calc_az_to_antisun(self):
        self._az_to_antisun = smallest_signed_angle(self.ra + np.pi, self.sun_ra)

    @property
    def az_to_sun(self):
        if self._az_to_sun is None:
            self.calc_az_to_sun()
        return self._az_to_sun

    @property
    def az_to_antisun(self):
        if self._az_to_antisun is None:
            self.calc_az_to_antisun()
        return self._az_to_antisun

    def set_auxtel_info(
        self,
        mjd,
        slewtime,
        sun_n18_setting,
        sun_n18_rising,
        moon_alt,
        moon_az,
        tel_alt_limits,
        tel_az_limits,
        sky_alt_limits,
        sky_az_limits,
        wind_speed,
        wind_direction,
        **kwargs,
    ):
        """Method to set all the information we expect will be required by
        a standard auxtel scheduler. Extra attributes can be set via **kwargs.

        Parameters
        ----------
        mjd : `float`
            Modified Julian Date (days).
        slewtime : `np.ndarray`, (N,)
            Healpix showing the slewtime to each healpixel center (seconds)
        moon_alt : `float`
            The altitude of the Moon (radians)
        moon_az : `float`
            The Azimuth of the moon (radians)
        wind_speed : `float`
            Wind speed (m/s).
        wind_direction : `float`
            Direction from which the wind originates. A direction of
            0.0 degrees means the wind originates from the north and 90.0
            degrees from the east (radians).
        sun_n18_setting : `float`
            The MJD when the sun is at -18 degrees altitude and setting
            during the current night. From interpolation.
        sun_n18_rising : `float`
            The MJD when the sun is at -18 degrees altitude and rising
            during the current night. From interpolation.
        sky_az_limits : `list` [[`float`, `float`]]
            A list of lists giving valid azimuth ranges. e.g.,
            [0, 2*np.pi] would mean all azimuth values are valid, while
            [[0, np.pi/2], [3*np.pi/2, 2*np.pi]] would mean anywhere in
            the south is invalid.  Radians.
        sky_alt_limits : `list` [[`float`, `float`]]
            A list of lists giving valid altitude ranges. Radians.
        tel_az_limits : `list` [`float`, `float`]
            A simple two-element list giving the valid azimuth ranges
            for the telescope movement. Radians.
        tel_alt_limits : `list` [`float`, `float`]
            A simple two-element list giving the valid altitude ranges
            for the telescope movement. Radians
        """
        self._init_attributes()
        auxtel_args = signature(self.set_auxtel_info)
        loc = locals()
        for key in auxtel_args.parameters.keys():
            setattr(self, key, loc[key])

        potential_attrs = dir(self)
        for key in kwargs:
            if key not in potential_attrs:
                warnings.warn("Setting unexpected Conditions attribute %s" % key)
            setattr(self, key, kwargs[key])

    def set_maintel_info(
        self,
        mjd,
        slewtime,
        current_filter,  # to become current_band
        mounted_filters,  # to become mounted_bands
        night,
        skybrightness,
        fwhm_eff,
        moon_alt,
        moon_az,
        moon_ra,
        moon_dec,
        moon_phase,
        sun_alt,
        sun_az,
        sun_ra,
        sun_dec,
        tel_ra,
        tel_dec,
        tel_alt,
        tel_az,
        wind_speed,
        wind_direction,
        sun_n12_setting,
        sun_n12_rising,
        sun_n18_setting,
        sun_n18_rising,
        moonrise,
        moonset,
        planet_positions,
        tel_alt_limits,
        tel_az_limits,
        sky_alt_limits,
        sky_az_limits,
        **kwargs,
    ):
        """Method to set all the information we expect will be required by
        a standard maintel scheduler. Extra attributes can be set via **kwargs.

        mjd : `float`
            Modified Julian Date (days).
        slewtime : `np.ndarray`, (N,)
            Healpix showing the slewtime to each healpixel center (seconds)
        current_filter : `str`
            The name of the current band. (expect one of u, g, r, i, z, y).
        mounted_filters : `list` [`str`]
            The bands that are currently mounted and thus available
            (expect 5 of u, g, r, i, z, y for LSSTCam).
        night : `int`
            The current night number (days). Probably starts at 1.
        skybrightness : `dict` {`str: `np.ndarray`, (N,)}
            Dictionary keyed by band name.
            Values are healpix arrays with the sky brightness at each
            healpix center (mag/acsec^2)
        fwhm_eff : `dict` {`str: `np.ndarray`, (N,)}
            Dictionary keyed by bandname.
            Values are the effective seeing FWHM at each healpix
            center (arcseconds)
        moon_alt : `float`
            The altitude of the Moon (radians)
        moon_az : `float`
            The Azimuth of the moon (radians)
        moon_ra : `float`
            RA of the moon (radians)
        moon_dec : `float`
            Declination of the moon (radians)
        moon_phase : `float`
            The Phase of the moon. (percent, 0=new moon, 100=full moon)
        sun_alt : `float`
            The altitude of the sun (radians).
        sun_az : `float`
            The Azimuth of the sun (radians).
        sun_ra : `float`
            The RA of the sun (radians).
        sun_dec : `float`
            The Dec of the sun (radians).
        tel_ra : `float`
            The current telescope RA pointing (radians).
        tel_dec : `float`
            The current telescope Declination (radians).
        tel_alt : `float`
            The current telescope altitude (radians).
        tel_az : `float`
            The current telescope azimuth (radians).
        wind_speed : `float`
            Wind speed (m/s).
        wind_direction : `float`
            Direction from which the wind originates. A direction of
            0.0 degrees means the wind originates from the north and 90.0
            degrees from the east (radians).
        sun_n12_setting : `float`
            The MJD of when the sun is at -12 degrees altitude and
            setting during the current night. From interpolation.
        sun_n18_setting : `float`
            The MJD when the sun is at -18 degrees altitude and setting
            during the current night. From interpolation.
        sun_n18_rising : `float`
            The MJD when the sun is at -18 degrees altitude and rising
            during the current night. From interpolation.
        sun_n12_rising : `float`
            The MJD when the sun is at -12 degrees altitude and rising
            during the current night. From interpolation.
        moonrise : `float`
            The MJD of moonrise during the current night. From interpolation.
        moonset : `float`
            The MJD of moon set during the current night. From interpolation.
        moon_phase_sunset : `float`
            The phase of the moon (0-100 illuminated) at sunset.
            Useful for setting which bands should be loaded.
        targets_of_opportunity : `list` [`rubin_scheduler.scheduler.targetoO`]
            targetoO objects.
        planet_positions : `dict` {`str`: `float`}
            Dictionary of planet name and coordinate e.g., 'venus_RA',
            'mars_dec'
        sky_az_limits : `list` [[`float`, `float`]]
            A list of lists giving valid azimuth ranges. e.g.,
            [0, 2*np.pi] would mean all azimuth values are valid, while
            [[0, np.pi/2], [3*np.pi/2, 2*np.pi]] would mean anywhere in
            the south is invalid.  Radians.
        sky_alt_limits : `list` [[`float`, `float`]]
            A list of lists giving valid altitude ranges. Radians.
        tel_az_limits : `list` [`float`, `float`]
            A simple two-element list giving the valid azimuth ranges
            for the telescope movement. Radians.
        tel_alt_limits : `list` [`float`, `float`]
            A simple two-element list giving the valid altitude ranges
            for the telescope movement. Radians
        """
        self._init_attributes()

        if night is not None:
            warnings.warn("Setting value for night no longer supported. Setting to None.")
            night = None

        maintel_args = signature(self.set_maintel_info)
        loc = locals()
        for key in maintel_args.parameters.keys():
            if key != "night":
                setattr(self, key, loc[key])

        potential_attrs = dir(self)
        for key in kwargs:
            if key not in potential_attrs:
                warnings.warn("Setting unexpected Conditions attribute %s" % key)
            if key != "night":
                setattr(self, key, kwargs[key])

    def set_attrs(self, **kwargs):
        """Convenience function for setting lots of attributes at once.
        See __init__ docstring for full list of potential attributes."""
        potential_attrs = dir(self)
        for key in kwargs:
            if key not in potential_attrs:
                warnings.warn("Setting unexpected Conditions attribute %s" % key)
            setattr(self, key, kwargs[key])

    def __repr__(self):
        return f"<{self.__class__.__name__} mjd='{self.mjd}' at {hex(id(self))}>"

    def __str__(self):
        # If dependencies of to_markdown are not installed, fall back on repr
        try:
            pd.DataFrame().to_markdown()
        except ImportError:
            return repr(self)

        if self.mjd is None:
            # Many elements of the pretty str representation rely on
            # the time beging set. If it is not, just return the ugly form.
            return repr(self)

        output = StringIO()
        print(f"{self.__class__.__qualname__} at {hex(id(self))}", file=output)
        print("============================", file=output)
        print("nside: ", self.nside, "  ", file=output)
        print("site: ", self.site.name, "  ", file=output)
        print("exptime: ", self.exptime, "  ", file=output)
        print("lmst: ", self.lmst, "  ", file=output)
        print("clouds: ", self.clouds, "  ", file=output)
        print("current_band: ", self.current_band, "  ", file=output)
        print("mounted_bands: ", self.mounted_bands, "  ", file=output)
        print("night: ", self.night, "  ", file=output)
        print("wind_speed: ", self.wind_speed, "  ", file=output)
        print("wind_direction: ", self.wind_direction, "  ", file=output)
        print(
            "len(scheduled_observations): ",
            len(self.scheduled_observations),
            "  ",
            file=output,
        )
        print(
            "len(queue): ",
            None if self.queue is None else len(self.queue),
            "  ",
            file=output,
        )
        print("moonPhase: ", self.moon_phase, "  ", file=output)
        print("bulk_cloud: ", self.bulk_cloud, "  ", file=output)
        print("targets_of_opportunity: ", self.targets_of_opportunity, "  ", file=output)
        print("cumulative_azimuth_rad: ", self.cumulative_azimuth_rad, "  ", file=output)

        positions = [
            {
                "name": "sun",
                "alt": self.sun_alt,
                "az": self.sun_az,
                "RA": self.sun_ra,
                "decl": self.sun_dec,
            }
        ]
        positions.append(
            {
                "name": "moon",
                "alt": self.moon_alt,
                "az": self.moon_az,
                "RA": self.moon_ra,
                "decl": self.moon_dec,
            }
        )
        for planet_name in ("venus", "mars", "jupiter", "saturn"):
            positions.append(
                {
                    "name": planet_name,
                    "RA": float(np.max(self.planet_positions[planet_name + "_RA"])),
                    "decl": float(np.max(self.planet_positions[planet_name + "_dec"])),
                }
            )
        positions.append(
            {
                "name": "telescope",
                "alt": np.max(self.tel_alt),
                "az": np.max(self.tel_az),
                "RA": np.max(self.tel_ra),
                "decl": np.max(self.tel_dec),
                "rot": np.max(self.rot_tel_pos),
            }
        )
        positions = pd.DataFrame(positions).set_index("name")
        print(file=output)
        print("Positions (radians)", file=output)
        print("-------------------", file=output)
        print(positions.to_markdown(), file=output)

        positions_deg = np.degrees(positions)
        print(file=output)
        print("Positions (degrees)", file=output)
        print("-------------------", file=output)
        print(positions_deg.to_markdown(), file=output)

        events = (
            "mjd",
            "sunset",
            "sun_n12_setting",
            "sun_n18_setting",
            "sun_n18_rising",
            "sun_n12_rising",
            "sunrise",
            "moonrise",
            "moonset",
            "sun_0_setting",
            "sun_0_rising",
        )
        event_rows = []
        for event in events:
            try:
                mjd = getattr(self, event)
                time = pd.to_datetime(float(mjd) + 2400000.5, unit="D", origin="julian")
                event_rows.append({"event": event, "MJD": mjd, "date": time})
            except AttributeError:
                pass

        event_df = pd.DataFrame(event_rows).set_index("event").sort_values(by="MJD")
        print("", file=output)
        print("Events", file=output)
        print("------", file=output)
        print(event_df.to_markdown(), file=output)

        map_stats = []
        for map_name in self.global_maps - set(["zeros_map", "nan_map"]):
            try:
                values = getattr(self, map_name)
                map_stats.append(
                    {
                        "map": map_name,
                        "nside": hp.npix2nside(len(values)),
                        "min": np.nanmin(values),
                        "max": np.nanmax(values),
                        "median": np.nanmedian(values),
                    }
                )
            except AttributeError:
                pass

        for base_map_name in self.by_band_maps:
            for band in "ugrizy":
                try:
                    values = getattr(self, base_map_name)[band]
                    map_name = f"{base_map_name}_{band}"
                    map_stats.append(
                        {
                            "map": map_name,
                            "nside": hp.npix2nside(len(values)),
                            "min": np.nanmin(values),
                            "max": np.nanmax(values),
                            "median": np.nanmedian(values),
                        }
                    )
                except AttributeError:
                    pass

        maps_df = pd.DataFrame(map_stats).set_index("map")
        print("", file=output)
        print("Maps", file=output)
        print("----", file=output)
        print(maps_df.to_markdown(), file=output)

        result = output.getvalue()
        return result

    def _repr_markdown_(self):
        return str(self)
