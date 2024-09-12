__all__ = ("Conditions",)

import functools
import warnings
from io import StringIO

import healpy as hp
import numpy as np
import pandas as pd

from rubin_scheduler.scheduler.utils import match_hp_resolution, set_default_nside, smallest_signed_angle
from rubin_scheduler.utils import (
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
        nside=None,
        site="LSST",
        exptime=30.0,
        mjd=None,
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

        Attributes (to be set by user/telemetry stream/scheduler)
        -------------------------------------------
        mjd : `float`
            Modified Julian Date (days).
        bulk_cloud : `float`
            The fraction of sky covered by clouds. (In the future might
            update to transparency map)
        cloud_map : `np.ndarray`, (N,)
            XXX--to be done. HEALpix array with cloudy pixels set to NaN.
        slewtime : `np.ndarray`, (N,)
            Healpix showing the slewtime to each healpixel center (seconds)
        current_filter : `str`
            The name of the current filter. (expect one of u, g, r, i, z, y).
        mounted_filters : `list` [`str`]
            The filters that are currently mounted and thus available
            (expect 5 of u, g, r, i, z, y for LSSTCam).
        night : `int`
            The current night number (days). Probably starts at 1.
        skybrightness : `dict` {`str: `np.ndarray`, (N,)}
            Dictionary keyed by filter name.
            Values are healpix arrays with the sky brightness at each
            healpix center (mag/acsec^2)
        fwhm_eff : `dict` {`str: `np.ndarray`, (N,)}
            Dictionary keyed by filtername.
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
        airmass : `np.ndarray`, (N,)
            A healpix map with the airmass value of each healpixel. (unitless)
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
        mjd_start : `float`
            The starting MJD of the survey.
        moonrise : `float`
            The MJD of moonrise during the current night. From interpolation.
        moonset : `float`
            The MJD of moon set during the current night. From interpolation.
        moon_phase_sunset : `float`
            The phase of the moon (0-100 illuminated) at sunset.
            Useful for setting which filters should be loaded.
        targets_of_opportunity : `list` [`rubin_scheduler.scheduler.targetoO`]
            targetoO objects.
        planet_positions : `dict` {`str`: `float`}
            Dictionary of planet name and coordinate e.g., 'venus_RA',
            'mars_dec'
        scheduled_observations : `np.ndarray`, (M,)
            A list of MJD times when there are scheduled observations.
            Defaults to empty array.
        tel_az_limits : `list` [[`float`, `float`]]
            A list of lists giving valid azimuth ranges. e.g.,
            [0, 2*np.pi] would mean all azimuth values are valid, while
            [[0, np.pi/2], [3*np.pi/2, 2*np.pi]] would mean anywhere in
            the south is invalid.  Radians.
        tel_alt_limits : `list` [[`float`, `float`]]
            A list of lists giving valid altitude ranges. Radians.
        altaz_limit_pad : `float`
            Pad to surround the tel_az_limits and tel_alt_limits with.

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
        pa : `np.ndarray`, (N,)
            The parallactic angle of each healpixel (radians). Recalculated
            if mjd is updated. Based on the fast approximate alt,az values.
        lmst : `float`
            The local mean sidereal time (hours). Updates is mjd is changed.
        m5_depth : `dict` {`str: `np.ndarray`, (N,)}
            the 5-sigma limiting depth healpix maps, keyed by filtername
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

        Attributes (set by the scheduler)
        -------------------------------
        queue : `list` [`observation` objects]
            The current queue of observations core_scheduler is waiting to
            execute.

        """
        if nside is None:
            nside = set_default_nside()
        self.nside = nside

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
        self.current_filter = None
        self.mounted_filters = None
        self.night = None
        self._lmst = None
        # Should be a dict with filtername keys
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

        # Telescope limits - these can be None
        # These should be in radians.
        self.tel_az_limits = None
        self.tel_alt_limits = None
        # Kinematic model (real slew) limits - these can't be None
        # if you are using the AltAzShadowMaskBasisFunction.
        # These generous limits won't restrict the AltAzShadowMask.
        # Radians.
        self.kinematic_alt_limits = [np.radians(-10), np.radians(100)]
        self.kinematic_az_limits = [np.radians(-250), np.radians(250)]
        # This has a (reasonable) default value, to avoid failure of
        # AltAzShadowMask in case this isn't set otherwise.
        self.altaz_limit_pad = np.radians(2)

        # Full sky cloud map
        self._cloud_map = None
        self._HA = None

        # XXX--document
        self.bulk_cloud = None

        self.rot_tel_pos = None

        self.targets_of_opportunity = None

        # Potential attributes that get computed
        self._solar_elongation = None
        self._az_to_sun = None
        self._az_to_antisun = None

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
    def cloud_map(self):
        return self._cloud_map

    @cloud_map.setter
    def cloud_map(self, value):
        self._cloud_map = match_hp_resolution(value, nside_out=self.nside)

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
        return self._airmass

    @airmass.setter
    def airmass(self, value):
        self._airmass = match_hp_resolution(value, nside_out=self.nside)
        self._m5_depth = None

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
    def m5_depth(self):
        if self._m5_depth is None:
            self.calc_m5_depth()
        return self._m5_depth

    def calc_m5_depth(self):
        self._m5_depth = {}
        for filtername in self._skybrightness:
            good = ~np.isnan(self._skybrightness[filtername])
            self._m5_depth[filtername] = self.nan_map.copy()
            self._m5_depth[filtername][good] = m5_flat_sed(
                filtername,
                self._skybrightness[filtername][good],
                self._fwhm_eff[filtername][good],
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
        print("current_filter: ", self.current_filter, "  ", file=output)
        print("mounted_filters: ", self.mounted_filters, "  ", file=output)
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
                "alt": self.tel_alt,
                "az": self.tel_az,
                "RA": self.tel_ra,
                "decl": self.tel_dec,
                "rot": self.rot_tel_pos,
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
            "mjd_start",
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
