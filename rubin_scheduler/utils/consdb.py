import importlib.util
import urllib.parse
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cache, cached_property
from math import log, sqrt
from warnings import warn

import healpy as hp
import numpy as np
import pandas as pd

# import rubin_sim
from astropy.coordinates import angular_separation
from astropy.coordinates.earth import EarthLocation
from astropy.time import Time

from rubin_scheduler.scheduler.utils import SchemaConverter
from rubin_scheduler.site_models import Almanac, SeeingModel
from rubin_scheduler.utils import (
    SURVEY_START_MJD,
    Site,
    SysEngVals,
    _angular_separation,
    _approx_altaz2pa,
    pseudo_parallactic_angle,
    rotation_converter,
)

GAUSSIAN_FWHM_OVER_SIGMA: float = 2.0 * sqrt(2.0 * log(2.0))
KNOWN_INSTRUMENTS = ["lsstcomcam", "lsstcomcamsim"]


def query_consdb(query: str, url: str = "postgresql://usdf@usdf-summitdb.slac.stanford.edu:5432/exposurelog"):
    """_summary_

    Parameters
    ----------
    query : `str`
        The SQL query to send to the consdb.
    url : `str`, optional
        The database connection string,
        by default "postgresql://usdf@usdf-summitdb.slac.stanford.edu:5432/exposurelog"

    Returns
    -------
    result : `pandas.DataFrame`
        The result of the query.
    """
    url_scheme: str = urllib.parse.urlparse(url).scheme
    match url_scheme:
        case "postgresql":
            can_support_postgresql: bool = bool(
                importlib.util.find_spec("sqlalchemy") and importlib.util.find_spec("psycopg2")
            )
            if not can_support_postgresql:
                raise RuntimeError("Optional dependencies required for postgresql access not installed")

            # import sqlalchemy here rather than at the top to keep mypy happy.
            # Add the type: ignore so IDEs do not complain when running in an
            # environment without it.
            import sqlalchemy  # type: ignore

            connection = sqlalchemy.create_engine(url)
            query_results: pd.DataFrame = pd.read_sql(query, connection)
        case "http" | "https":
            can_support_http: bool = bool(
                importlib.util.find_spec("lsst")
                and importlib.util.find_spec("lsst.summit")
                and importlib.util.find_spec("lsst.summit.utils")
            )
            if not can_support_http:
                raise RuntimeError("Optional dependencies required for ConsDB access not installed")

            # import ConsDbClient here rather than at the top to satisfy mypy.
            # Add the type: ignore so IDEs do not complain when running in an
            # environment without it.
            from lsst.summit.utils import ConsDbClient  # type: ignore

            consdb = ConsDbClient(url)
            query_results: pd.DataFrame = consdb.query(query).to_pandas()
        case _:
            raise ValueError(f"Unrecongined url scheme {url_scheme} in consdb url {url}")

    return query_results


# Different instruments (and different numbers of snaps per visit) do or
# will have differenc schema in the consdb, so make a base class with
# expected common functionality, and use derived classes for
# instrument-specific elements.

# Separate different pieces of metadata (columns in a table of visits) into
# different methods to make it easy to individually override them by
# instrument.


# cached_properties are a way to compute values when and only when needed
# but these won't update if the date is changed. In my expected use, that's
# fine; the user should just instantiate another ConsDBVisits instance if
# they want a different database or date. But, what if someone does?
# To avoid this, make it a frozen dataclass, which prevents members from
# being updated.
@dataclass(frozen=True)
class ConsDBVisits(ABC):
    """Visits returned from a query to the consdb.

    Parameters
    ----------
    day_obs : `str`
        The date for which to query the database.
    url : `str`
        The connection string from the database.
    num_nights : `int`
        The number of nights (up to and including day_obs)
        for which to get visits. Defaults to 1.
    """

    day_obs: str | int
    url: str = "postgresql://usdf@usdf-summitdb.slac.stanford.edu:5432/exposurelog"
    num_nights: int = 1

    @property
    @abstractmethod
    def instrument(self) -> str:
        """The instrument.

        Returns
        -------
        instrument : `str`
            The instrument.
        """
        # Subclasses should return a string constant specifying the relevant
        # instrument.
        # Specified as an property rather than an attribute so it can be
        # abstract.
        raise NotImplementedError

    @property
    @abstractmethod
    def num_exposures(self) -> int:
        """The number of snaps per visit.

        Returns
        -------
        num_exposures : `int`
            The number of snaps per visit.
        """
        # Subclasses should return a string constant specifying the relevant
        # instrument.
        # Specified as an property rather than an attribute so it can be
        # abstract.
        raise NotImplementedError

    @property
    @abstractmethod
    def telescope(self) -> str:
        """The telescope.

        Returns
        -------
        telescope : `str`
            The telescope.
        """
        # Subclasses should return a constant specifying the relevant
        # instrument.
        # Specified as an property rather than an attribute so it can be
        # abstract.
        raise NotImplementedError

    @property
    @abstractmethod
    def gain(self) -> pd.Series:
        """The gain.

        Returns
        -------
        gain : `pd.Series`
            The gain.
        """
        # Subclasses should return a constant specifying the relevant
        # instrument.
        # Specified as an property rather than an attribute so it can be
        # abstract.
        raise NotImplementedError

    @property
    @abstractmethod
    def pixel_scale(self) -> pd.Series:
        """The pixel scale.

        Returns
        -------
        pixel_scale : `pd.Series`
            The pixel scale.
        """
        # Subclasses should return a constant specifying the relevant
        # instrument.
        # Specified as an property rather than an attribute so it can be
        # abstract.
        raise NotImplementedError

    @abstractmethod
    def instrumental_zeropoint_for_band(self, band: str) -> float:
        """The zero point.

        Parameters
        ----------
        band : `str`
            The band for which to return the zero point.

        Returns
        -------
        zero_point : `float`
            The photometric zero point.
        """
        # Subclasses should return a constant specifying the relevant
        # instrument.
        # Specified as an property rather than an attribute so it can be
        # abstract.
        raise NotImplementedError

    @property
    @abstractmethod
    def site(self) -> Site:
        """The site.

        Returns
        -------
        site : `rubin_scheduler.utils.site.Site`
            The observatory site.
        """
        # Subclasses should return a constant specifying the relevant
        # instrument.
        # Specified as an property rather than an attribute so it can be
        # abstract.
        raise NotImplementedError

    @cached_property
    def almanac(self) -> Almanac:
        """The astronomical almanac.

        Returns
        -------
        almanac : `rubin_scheduler.site_models.almanac.Almanac`
            Astronomical almanac data.
        """
        return Almanac()

    @cached_property
    def location(self) -> EarthLocation:
        """The observatory location.

        Returns
        -------
        location : `astropy.coordinates.earth.EarthLocation`
            The observatory location.
        """
        #
        return EarthLocation(lat=self.site.latitude, lon=self.site.longitude, height=self.site.height)

    @cached_property
    def day_obs_time(self) -> Time:
        """The day_obs as an astropy time.

        Returns
        -------
        day_obs : `astropy.time.Time`
            The astropy time representation of day_obs.
        """
        time: Time = Time(self.day_obs)
        return time

    @cached_property
    def day_obs_mjd(self) -> int:
        """The day_obs as an mjd.

        Returns
        -------
        day_obs_mjd : `int`
            The mjd representation of day_obs.
        """
        # make mypy happy
        assert isinstance(self.day_obs_time.mjd, float)
        return int(self.day_obs_time.mjd)

    @cached_property
    def day_obs_int(self) -> int:
        """The day_obs as an integer.

        Returns
        -------
        day_obs_int : `int`
            The integer representation of day_obs.
        """
        # make mypy happy
        assert isinstance(self.day_obs_time.iso, str)
        return int(self.day_obs_time.iso[:10].replace("-", ""))

    @cached_property
    def consdb_visits(self) -> pd.DataFrame:
        """The visit data as returned from the consdb

        Returns
        -------
        consdb_visits : `pd.DataFrame`
            The visit data.
        """
        day_obs_date = datetime.strptime(f"{self.day_obs_int}", "%Y%m%d").date()
        prior_day_obs_date = day_obs_date - timedelta(days=self.num_nights)
        prior_day_obs_int = int(prior_day_obs_date.strftime("%Y%m%d"))

        # To avoid duplicate columns, explicitly include list of columns to
        # to return in the query instead of using a "SELECT *".
        def columns_query(table, instrument):
            return f"""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = '{table}'
                  AND table_schema = 'cdb_{instrument}'
            """

        visit_columns = (
            query_consdb(columns_query(f"visit{self.num_exposures}", self.instrument))
            .loc[:, "column_name"]
            .values
        )
        ql_columns = (
            query_consdb(columns_query(f"visit{self.num_exposures}_quicklook", self.instrument))
            .loc[:, "column_name"]
            .values
        )
        visit_query_visit_columns = ", ".join([f"v.{c} AS {c}" for c in visit_columns])
        visit_query_ql_columns = ", ".join(
            [f"v{self.num_exposures}q.{c} AS {c}" for c in ql_columns if c not in visit_columns]
        )
        visit_query_columns = ", ".join([visit_query_visit_columns, visit_query_ql_columns])

        consdb_visits_query: str = f"""
            SELECT {visit_query_columns}
            FROM cdb_{self.instrument}.visit1 AS v
            LEFT JOIN cdb_{self.instrument}.visit{self.num_exposures}_quicklook
                AS v{self.num_exposures}q ON v.visit_id = v{self.num_exposures}q.visit_id
            WHERE v.obs_start_mjd IS NOT NULL
                AND v.s_ra IS NOT NULL
                AND v.s_dec IS NOT NULL
                AND v.sky_rotation IS NOT NULL
                AND ((v.band IS NOT NULL) OR (v.physical_filter IS NOT NULL))
                AND v.day_obs <= {self.day_obs_int}
                AND v.day_obs > {prior_day_obs_int}
        """
        return query_consdb(consdb_visits_query, self.url)

    @cached_property
    def visit_id(self) -> pd.Series:
        """The unique identifier from the consdb database.

        Returns
        -------
        visit_id : `pd.Series`
            The unique identifier from the consdb database.
        """
        return self.consdb_visits["visit_id"]

    @cached_property
    def observation_id(self) -> pd.Series:
        """The unique identifier from the consdb database.

        Returns
        -------
        observation_id : `pd.Series`
            The unique identifier from the consdb database.
        """
        # Instead of just using visit_id, create a separate property
        # that can be overridden by subclasses which might not get a valid
        # visit_id.
        return self.visit_id

    @cached_property
    def ra(self) -> pd.Series:
        """The R.A. of each visit, in degrees.

        Returns
        -------
        ra : `pd.Series`
            The R.A. of each visit, in degrees.
        """
        return self.consdb_visits["s_ra"]

    @cached_property
    def decl(self) -> pd.Series:
        """The declination if each visit, in degrees.

        Returns
        -------
        decl : `pd.Series`
            The declination of each visit, in degrees.
        """
        return self.consdb_visits["s_dec"]

    @cached_property
    def obs_start_mjd(self) -> pd.Series:
        """The MJD at the start of each visit.

        Returns
        -------
        obs_start_mjd : `pd.Series`
            The MJD at the start of each visit.
        """
        return self.consdb_visits["obs_start_mjd"]

    @cached_property
    def shut_time(self) -> pd.Series:
        """Spatially-averaged shutter-open duration of each visit.

        Returns
        -------
        shut_time : `pd.Series`
            Spatially-averaged shutter-open duration.
        """
        return self.consdb_visits["shut_time"]

    @cached_property
    def obs_midpt_mjd(self) -> pd.Series:
        """Midpoint time for exposure at the fiducial center of the focal
        plane. array in MJD.

        Returns
        -------
        midpt_mjd: `pd.Series`
            Midpoint time for exposure at the fiducial center of the focal
            plane. array in MJD.
        """
        return self.consdb_visits["obs_midpt_mjd"]

    @cached_property
    def sky_rotation(self) -> pd.Series:
        """Targeted sky rotation angle.

        Returns
        -------
        sky_rotation : `pd.Series`
            Targeted sky rotation angle for each visit.
        """
        return self.consdb_visits["sky_rotation"]

    @cached_property
    def target_name(self) -> pd.Series:
        """The target name for each visit.

        Returns
        -------
        target_name : `pd.Series`
            The target name for each visit.
        """
        return self.consdb_visits["target_name"]

    @cached_property
    def airmass(self) -> pd.Series:
        """Airmass of each visit.

        Returns
        -------
        airmass : `pd.Series`
            Airmass of each visit.
        """
        return self.consdb_visits["airmass"]

    @cached_property
    def band(self) -> pd.Series:
        """Abstract filter for each visit

        Returns
        -------
        band : `pd.Series`
            Abstract filter for each visit.
        """
        return self.consdb_visits["band"]

    @cached_property
    def azimuth(self) -> pd.Series:
        """Azimuth for each visit (degrees) at central time.

        Returns
        -------
        azimuth : `pd.Series`
            Azimuth for each visit.
        """
        return self.consdb_visits["azimuth"]

    @cached_property
    def altitude(self) -> pd.Series:
        """Altitude for each visit (degrees) at central time.

        Returns
        -------
        altitude : `pd.Series`
            Altitude for each visit (degrees).
        """
        return self.consdb_visits["altitude"]

    @cached_property
    def zd(self) -> pd.Series:
        """Zenith distance for each visit at central time (degrees)

        Returns
        -------
        zd : `pd.Series`
            Zenith distance for each visit at central time (degrees)
        """
        return 90 - self.altitude

    @cached_property
    def note(self) -> pd.Series:
        """Note for each visit.

        Returns
        -------
        note : `pd.Series`
            Note for each visit.
        """
        if "note" in self.consdb_visits:
            note = self.consdb_visits["note"]
        else:
            note = pd.Series("", index=self.consdb_visits.index)
        return note

    @cached_property
    def scheduler_note(self) -> pd.Series:
        """scheduler_note for each visit.

        Returns
        -------
        note : `pd.Series`
            Note for each visit.
        """
        if "scheduler_note" in self.consdb_visits:
            note = self.consdb_visits["scheduler_note"]
        else:
            note = pd.Series("", index=self.consdb_visits.index)
        return note

    @cached_property
    def observation_reason(self) -> pd.Series:
        """Observation reason for each visit.

        Returns
        -------
        observation_reason : `pd.Series`
            Observation reason for each visit.
        """
        if "observation_reason" in self.consdb_visits:
            reason = self.consdb_visits["observation_reason"]
        else:
            reason = pd.Series("", index=self.consdb_visits.index)
        return reason

    @cached_property
    def science_program(self) -> pd.Series:
        """Science program each visit.

        Returns
        -------
        science_progarm : `pd.Series`
            Science program for each visit.
        """
        if "science_program" in self.consdb_visits:
            science_program = self.consdb_visits["science_program"]
        else:
            science_program = pd.Series("", index=self.consdb_visits.index)
        return science_program

    @cached_property
    def visit_time(self) -> pd.Series:
        """Visit time for each visit (seconds).

        Returns
        -------
        visit_time : `pd.Series`
            Visit time for each visit.
        """
        return (self.consdb_visits["obs_end_mjd"] - self.consdb_visits["obs_start_mjd"]) * 24 * 60 * 60

    @cached_property
    def cloud(self) -> pd.Series:
        """Fractional cloud cover of the sky for each visit.

        Returns
        -------
        cloud : `pd.Series`
            Fractional cloud cover of the sky for each visit.
        """
        return pd.Series(np.nan, index=self.consdb_visits.index)

    @cached_property
    def slew_distance(self) -> pd.Series:
        """Slew distance to reach each visit (degrees).

        Returns
        -------
        slew_distance : `pd.Series`
            Slew distance to reach each visit (degrees).
        """
        slew_distance = pd.Series(np.nan, index=self.consdb_visits.index)
        previous_visit_idx = None
        for visit_idx in self.consdb_visits.index:
            if previous_visit_idx is not None:
                this_slew_distance = angular_separation(
                    np.radians(self.ra[previous_visit_idx]),
                    np.radians(self.decl[previous_visit_idx]),
                    np.radians(self.ra[visit_idx]),
                    np.radians(self.decl[visit_idx]),
                )
                slew_distance[visit_idx] = np.degrees(this_slew_distance)
            previous_visit_idx = visit_idx
        return slew_distance

    @cached_property
    def start_time(self) -> Time:
        """Start time for each visit.

        Returns
        -------
        start_time : `Time`
            Start time for each visit.
        """
        return Time(self.obs_start_mjd, format="mjd")

    @cached_property
    def observation_start_lst(self) -> pd.Series:
        """Apparent local sidereal time (degrees) at the start of each visit.

        Returns
        -------
        lst : `pd.Series`
            Apparent local sidereal time (degrees) at the start of each visit.
        """
        start_lst = pd.Series(
            self.start_time.sidereal_time("apparent", self.location).degree, index=self.consdb_visits.index
        )
        return start_lst

    @cached_property
    def pseudo_parallactic_angle(self) -> pd.Series:
        """Parallactic angle as defined in SMTN-019 for each visit (degrees).

        Returns
        -------
        pa : `pd.Series`
            Parallactic angle as defined in SMTN-019 for each visit.
        """
        # Following sim_runner
        # Using pseudo_parallactic_angle, see https://smtn-019.lsst.io/v/DM-44258/index.html

        # make mypy happy
        assert isinstance(self.decl.values, np.ndarray)
        assert isinstance(self.ra.values, np.ndarray)
        assert isinstance(self.obs_start_mjd.values, np.ndarray)
        assert isinstance(self.site.longitude, float)
        assert isinstance(self.site.latitude, float)
        assert isinstance(self.site.height, float)

        pa, alt, az = pseudo_parallactic_angle(
            self.ra.values,
            self.decl.values,
            self.obs_start_mjd.values,
            lon=self.site.longitude,
            lat=self.site.latitude,
            height=self.site.height,
        )
        return pa

    @cached_property
    def rot_tel_pos(self) -> pd.Series:
        """Telescope rotator position (degrees).

        Returns
        -------
        rot_tel_pos : pd.Series
            Telescope rotator position (degrees).
        """
        rot_tel_pos = rotation_converter(telescope=self.telescope)._rotskypos2rottelpos(
            np.radians(self.sky_rotation), np.radians(self.pseudo_parallactic_angle)
        )
        return np.degrees(rot_tel_pos)

    @cached_property
    def parallactic_angle(self) -> pd.Series:
        """Parallactic angle (degrees).

        Returns
        -------
        pa : `pd.Series`
            Parallactic angle (degrees).
        """
        # Following sim_runner
        pa = pd.Series(
            np.degrees(
                _approx_altaz2pa(np.radians(self.altitude), np.radians(self.azimuth), self.site.latitude_rad)
            ),
            index=self.consdb_visits.index,
        )
        return pa

    @cached_property
    def night(self) -> int:
        """The night of the survey.

        Returns
        -------
        night : `int`
            The night of the survey.
        """
        night: int = 1 + self.day_obs_mjd - int(np.floor(SURVEY_START_MJD))
        return night

    @cached_property
    def sun_moon_positions(self) -> pd.DataFrame:
        """Sun and moon R.A., Declination, Azimuth, and Altitude (degrees)
        at each visit.

        Returns
        -------
        positions : `pd.DataFrame`
            Sun and moon positions (degrees).
        """
        sun_moon = pd.DataFrame(
            self.almanac.get_sun_moon_positions(self.obs_midpt_mjd.values), index=self.consdb_visits.index
        )

        # Convert to degrees, because everything else is in degrees
        for body in ("sun", "moon"):
            for coord in ("alt", "az", "RA", "dec"):
                column = f"{body}_{coord}"
                sun_moon[column] = np.degrees(sun_moon[column])

        return sun_moon

    @cached_property
    def moon_distance(self) -> pd.Series:
        """Angle between each visit pointing the moon (degrees).

        Returns
        -------
        distance : `pd.Series`
            Angle between each visit pointing the moon (degrees).
        """
        # make mypy happy
        assert isinstance(self.decl.values, np.ndarray)
        assert isinstance(self.ra.values, np.ndarray)

        moon_ra = self.sun_moon_positions["moon_RA"].values
        moon_decl = self.sun_moon_positions["moon_dec"].values
        assert isinstance(moon_ra, np.ndarray)
        assert isinstance(moon_decl, np.ndarray)

        moon_distance_rad = _angular_separation(
            np.radians(self.ra.values),
            np.radians(self.decl.values),
            np.radians(moon_ra),
            np.radians(moon_decl),
        )
        return pd.Series(np.degrees(moon_distance_rad), index=self.consdb_visits.index)

    @cached_property
    def solar_elong(self) -> pd.Series:
        """Solar elongation (degrees) of each visit.

        Returns
        -------
        elong : `pd.Series`
            Solar elongation of each visit in degrees.
        """
        # make mypy happy
        assert isinstance(self.decl.values, np.ndarray)
        assert isinstance(self.ra.values, np.ndarray)

        sun_ra = self.sun_moon_positions["sun_RA"].values
        sun_decl = self.sun_moon_positions["sun_dec"].values
        assert isinstance(sun_ra, np.ndarray)
        assert isinstance(sun_decl, np.ndarray)

        solar_elong_rad = _angular_separation(
            np.radians(self.ra.values),
            np.radians(self.decl.values),
            np.radians(sun_ra),
            np.radians(sun_decl),
        )
        return pd.Series(np.degrees(solar_elong_rad), index=self.consdb_visits.index)

    @cached_property
    def zeropoint_e(self) -> pd.Series:
        """Photometric zero point for each visit, with "counts" in electons.

        Returns
        -------
        zp : `pd.Series`
            Photometric zero point for each visit, with "counts" in electons.
        """
        return self.consdb_visits["zero_point_median"] + 2.5 * np.log10(self.gain)

    @cached_property
    def instrumental_zeropoint_e(self) -> pd.Series:
        """Instrumental zero point for each visit, with "counts" in electons.

        Returns
        -------
        zp : `pd.Series`
            Instrumental zero point for each visit, with "counts" in electons.
        """
        return self.band.apply(self.instrumental_zeropoint_for_band) + 2.5 * np.log10(self.shut_time)

    @cached_property
    def sky_e_per_pixel(self) -> pd.Series:
        """Median sky background of each visit (in electons).

        Returns
        -------
        sky : `pd.Series`
            Median sky background of each visit (in electons).
        """
        return self.consdb_visits["sky_bg_median"] * self.gain

    @cached_property
    def sky_mag_per_asec(self) -> pd.Series:
        """Median sky background in each visit, in mags asec^-2

        Returns
        -------
        sky : `pd.Series`
            Median sky background in each visit, in mags asec^-2
        """
        return self.instrumental_zeropoint_e - 2.5 * np.log10(self.sky_e_per_pixel / (self.pixel_scale**2))

    @cache
    def hpix(self, nside: int) -> pd.Series:
        """The heaplix of the pointing for each visit, for a given nside.

        Parameters
        ----------
        nside : `int`
            The nside of the healpix map to use.

        Returns
        -------
        hpix : `pd.Series`
            The heaplix of the pointing for each visit, for a given nside.
        """
        return pd.Series(hp.ang2pix(nside, self.ra, self.decl, lonlat=True), index=self.consdb_visits.index)

    @cached_property
    def fwhm_eff(self) -> pd.Series:
        """Effective PSF FWHM (arcseconds).

        Returns
        -------
        fwhm : `pd.Series`
            Effective PSF FWHM (arcseconds).
        """
        return self.consdb_visits["psf_sigma_median"] * GAUSSIAN_FWHM_OVER_SIGMA * self.pixel_scale

    @cached_property
    def fwhm_geom(self) -> pd.Series:
        """Geometric PSF FWHM (arcseconds).

        Returns
        -------
        fwhm : `pd.Series`
            Geometric PSF FWHM (arcseconds).
        """
        return SeeingModel.fwhm_eff_to_fwhm_geom(self.fwhm_eff)

    @cached_property
    def fwhm_500(self) -> pd.Series:
        """FWHM at zenith and 500nm, in arcseconds, at the time of each visit.

        Returns
        -------
        fwhm : `pd.Series`
            FWHM at zenith and 500nm, in arcseconds, at the time of each visit.
        """
        return self.consdb_visits["seeing_zenith_500nm_median"] * GAUSSIAN_FWHM_OVER_SIGMA * self.pixel_scale

    @cached_property
    def eff_time_m5(self) -> pd.Series:
        """Effective exposure time, in seconds.
        This is the time to reach the m5 depeth in each visit, under
        a reference set of conditions.

        Returns
        -------
        eff_time : `pd.Series`
            The effective exposure time.
        """
        return self.consdb_visits["eff_time_m5"]

    @cached_property
    def opsim(self) -> pd.DataFrame:
        """The table of visits, in a format replicating opsim output.

        Returns
        -------
        opsim : `pd.DataFrame`
            The table of visits, in a format replicating opsim output.
        """

        opsim_dict = {
            "observationId": self.observation_id,
            "fieldRA": self.ra,
            "fieldDec": self.decl,
            "observationStartMJD": self.obs_start_mjd,
            "flush_by_mjd": np.nan,
            "visitExposureTime": self.shut_time,
            "filter": self.band,
            "rotSkyPos": self.sky_rotation,
            "rotSkyPos_desired": np.nan,
            "numExposures": self.num_exposures,
            "airmass": self.airmass,
            "seeingFwhm500": self.fwhm_500,
            "seeingFwhmEff": self.fwhm_eff,
            "seeingFwhmGeom": self.fwhm_geom,
            "skyBrightness": self.sky_mag_per_asec,
            "night": self.night,
            "slewTime": np.nan,
            "visitTime": self.visit_time,
            "slewDistance": self.slew_distance,
            "fiveSigmaDepth": self.eff_time_m5,
            "altitude": self.altitude,
            "azimuth": self.azimuth,
            "paraAngle": self.parallactic_angle,
            "psudoParaAngle": self.pseudo_parallactic_angle,
            "cloud": self.cloud,
            "moonAlt": self.sun_moon_positions["moon_alt"],
            "sunAlt": self.sun_moon_positions["sun_alt"],
            "note": self.note,
            "scheduler_note": self.scheduler_note,
            "target_name": self.target_name,
            #            "block_id": "NOT AVAILABLE",
            "observationStartLST": self.observation_start_lst,
            "rotTelPos": self.rot_tel_pos,
            "rotTelPos_backup": np.nan,
            "moonAz": self.sun_moon_positions["moon_az"],
            "sunAz": self.sun_moon_positions["sun_az"],
            "sunRA": self.sun_moon_positions["sun_RA"],
            "sunDec": self.sun_moon_positions["sun_dec"],
            "moonRA": self.sun_moon_positions["moon_RA"],
            "moonDec": self.sun_moon_positions["moon_dec"],
            "moonDistance": self.moon_distance,
            "solarElong": self.solar_elong,
            "moonPhase": self.sun_moon_positions["moon_phase"],
            "cummTelAz": np.nan,
            #            "scripted_id": "NOT AVAILABLE",
            "observation_reason": self.observation_reason,
            "science_program": self.science_program,
            "start_date": self.consdb_visits["obs_start"],
        }

        opsim_df = pd.DataFrame(opsim_dict)

        return opsim_df

    @cached_property
    def obs(self) -> np.recarray:
        """The table of visits, in a format replicating an observation array.

        Returns
        -------
        obs : `np.recarray`
            The observation array.
        """
        return SchemaConverter().opsimdf2obs(self.opsim)

    @cached_property
    def merged_opsim_consdb(self):
        merged_visits = self.consdb_visits.merge(
            self.opsim, left_on="visit_id", right_on="observationId", suffixes=(None, "_opsim")
        )
        return merged_visits


class ComcamConsDBVisits(ConsDBVisits):

    def _have_numeric_values(self, column) -> bool:
        if column not in self.consdb_visits.columns:
            return False

        # Called to see if we need to guess values because the
        # consdb doesn't yet fill in what we need
        value_dtype = self.consdb_visits[column].dtype

        # Needed to make mypy happy
        assert isinstance(value_dtype, np.dtype)

        have_values = np.issubdtype(value_dtype, np.number)
        if not have_values:
            warn(f"Consdb does not have values for {column}, guessing values instead")

        return np.issubdtype(value_dtype, np.number)

    # Set "constant" properties using cached_property rather
    # than actual attributes, because abstract "members" can only be declared
    # as such using properties, and overriding a property with an actual
    # attribute confuses type checkers.

    @property
    def instrument(self) -> str:
        return "lsstcomcam"

    @property
    def num_exposures(self) -> int:
        return 1

    @property
    def telescope(self) -> str:
        return "rubin"

    @property
    def gain(self) -> pd.Series:
        return pd.Series(1.67, index=self.consdb_visits.index)

    @property
    def pixel_scale(self) -> pd.Series:
        return pd.Series(0.2, index=self.consdb_visits.index)

    def instrumental_zeropoint_for_band(self, band: str) -> float:
        zp_t = defaultdict(np.array(np.nan).item, SysEngVals().zp_t.items())
        return zp_t[band]

    @property
    def site(self):
        return Site("LSST")

    @cached_property
    def observation_id(self) -> pd.Series:
        """The unique identifier from the consdb database.

        Returns
        -------
        visit_id : `pd.Series`
            The unique identifier from the consdb database.
        """
        if self._have_numeric_values("visit_id"):
            observation_id = self.consdb_visits["visit_id"]
        elif self._have_numeric_values("day_obs") and self._have_numeric_values("seq_num"):
            observation_id = self.consdb_visits["day_obs"] * 10000 + self.consdb_visits["seq_num"]
            warn("Cannot use visit_id as observation_id, guessing instead!")
        else:
            warn("Can use neither visit_id nor guess observation_id, just making something up!")
            observation_id = pd.Series(
                Time(self.consdb_visits.obs_start).strftime("9%Y%m%d%H%M%S").astype(int),
                index=self.consdb_visits.index,
            )

        return observation_id

    @cached_property
    def azimuth(self) -> pd.Series:
        if self._have_numeric_values("azimuth"):
            azimuth = super().azimuth
        elif self._have_numeric_values("azimuth_start"):
            if self._have_numeric_values("azimuth_end"):
                azimuth = (self.consdb_visits["azimuth_start"] + self.consdb_visits["azimuth_end"]) / 2
            else:
                azimuth = self.consdb_visits["azimuth_start"]
        else:
            azimuth = pd.Series(np.nan, index=self.consdb_visits.index)

        return azimuth

    @cached_property
    def altitude(self) -> pd.Series:
        if self._have_numeric_values("altitude"):
            altitude = super().altitude
        elif self._have_numeric_values("altitude_start"):
            if self._have_numeric_values("altitude_end"):
                altitude = (self.consdb_visits["altitude_start"] + self.consdb_visits["altitude_end"]) / 2
            else:
                altitude = self.consdb_visits["altitude_start"]
        else:
            altitude = pd.Series(np.nan, index=self.consdb_visits.index)

        return altitude

    @cached_property
    def airmass(self) -> pd.Series:
        if self._have_numeric_values("airmass"):
            airmass = super().airmass
        else:
            # Calculate airmass using Cassini's model.
            # Models the atmosphere with a uniform spherical shell with a
            # height 1/470 of the radius of the earth.
            # Described in Kirstensen https://doi.org/10.1002/asna.2123190313
            # and https://doi.org/10.2172/1574836
            a_cos_zd = 470 * np.cos(np.radians(self.zd))
            airmass = pd.Series(np.sqrt(a_cos_zd**2 + 941) - a_cos_zd, index=self.consdb_visits.index)

        return airmass

    @cached_property
    def shut_time(self) -> pd.Series:
        if self._have_numeric_values("shut_time"):
            shut_time = super().shut_time
        else:
            shut_time = pd.Series(SysEngVals().exptime, index=self.consdb_visits.index)

        return shut_time

    @cached_property
    def obs_midpt_mjd(self) -> pd.Series:
        if self._have_numeric_values("obs_midpt_mjd"):
            mjd = super().obs_midpt_mjd
        elif self._have_numeric_values("obs_end_mjd"):
            mjd = (self.obs_start_mjd + self.consdb_visits["obs_end_mjd"]) / 2
        else:
            mjd = self.obs_start_mjd + (0.5 * self.shut_time) / 86400

        return mjd

    @cached_property
    def eff_time_m5(self):
        if self._have_numeric_values("eff_time_m5"):
            eff_time = super().eff_time_m5
        elif self._have_numeric_values("eff_time_median"):
            # From SMTN-002, and map to np.nan if not a standard band.
            ref_mags = defaultdict(
                np.array(np.nan).item,
                {"u": 23.70, "g": 24.97, "r": 24.52, "i": 24.13, "z": 23.56, "y": 22.55}.items(),
            )
            eff_time = self.band.map(ref_mags) + 1.25 * np.log10(self.consdb_visits["eff_time_median"] / 30)
        else:
            eff_time = pd.Series(np.nan, index=self.consdb_visits.index)

        return eff_time

    @cached_property
    def band(self) -> pd.Series:
        band = self.consdb_visits["band"].copy()
        missing_band: pd.Series[bool] = band.isna()
        band[missing_band] = self.consdb_visits.loc[missing_band, "physical_filter"].str.get(0)
        return band

    @cached_property
    def sky_e_per_pixel(self) -> pd.Series:
        """Median sky background of each visit (in electons).

        Returns
        -------
        sky : `pd.Series`
            Median sky background of each visit (in electons).
        """
        if self._have_numeric_values("sky_bg_median"):
            sky = self.consdb_visits["sky_bg_median"] * self.gain
        else:
            sky = pd.Series(np.nan, index=self.consdb_visits.index)
        return sky


class ComcamSimConsDBVisits(ComcamConsDBVisits):

    @property
    def instrument(self) -> str:
        return "lsstcomcamsim"


# Different subclasses of ConsDBOpsimConverter are needed for
# different instruments.
# Make a factory function to return the correct one.
def load_consdb_visits(instrument: str = "lsstcomcam", *args, **kwargs) -> ConsDBVisits:
    """Return visits from the consdb.

    Parameters
    ----------
    instrument : `str`, optional
        The name of the instrument, by default "lsstcomcamsim"

    Returns
    -------
    visits : `ConsDBVisits`
        Visits from the consdb.
    """

    if instrument not in KNOWN_INSTRUMENTS:
        # Yes, the "case: _" clause below would take care of this,
        # but having this explicitly here makes sure KNOWN_INSTRUMENTS
        # gets properly updated when new instruments are added.
        raise NotImplementedError

    match instrument:
        case "lsstcomcam":
            converter: ConsDBVisits = ComcamConsDBVisits(*args, **kwargs)
        case "lsstcomcamsim":
            converter: ConsDBVisits = ComcamSimConsDBVisits(*args, **kwargs)
        case _:
            raise NotImplementedError

    return converter
