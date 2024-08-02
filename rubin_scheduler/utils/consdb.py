import importlib.util
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache, cached_property
from math import log, sqrt

import healpy as hp
import numpy as np
import pandas as pd

# import rubin_sim
from astropy.coordinates import angular_separation
from astropy.coordinates.earth import EarthLocation
from astropy.time import Time

from rubin_scheduler.site_models import Almanac, SeeingModel
from rubin_scheduler.skybrightness_pre import SkyModelPre
from rubin_scheduler.utils import (
    Site,
    SysEngVals,
    _angular_separation,
    _approx_altaz2pa,
    pseudo_parallactic_angle,
    rotation_converter,
    survey_start_mjd,
)

GAUSSIAN_FWHM_OVER_SIGMA: float = 2.0 * sqrt(2.0 * log(2.0))


def query_consdb(query: str, url: str = "postgresql://usdf@usdf-summitdb.slac.stanford.edu:5432/exposurelog"):
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


# cached_properties are a great way to compute values when and only when needed
# but these won't update if the date is changed. In my expected use, that's
# fine; the user should just instantiate another ConsDBVisits instance if
# they want a different database or date. But, what if someone does?
# To avoid this, make it a frozen dataclass, which prevents members from
# being updated.
@dataclass(frozen=True)
class ConsDBVisits(ABC):

    day_obs: str | int
    url: str = "postgresql://usdf@usdf-summitdb.slac.stanford.edu:5432/exposurelog"

    @property
    @abstractmethod
    def instrument(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_exposures(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def telescope(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def gain(self) -> pd.Series:
        raise NotImplementedError

    @property
    @abstractmethod
    def pixel_scale(self) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def instrumental_zeropoint_for_band(self, band: str) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def site(self):
        raise NotImplementedError

    @cached_property
    def almanac(self):
        return Almanac()

    @cached_property
    def location(self):
        return EarthLocation(lat=self.site.latitude, lon=self.site.longitude, height=self.site.height)

    @cached_property
    def day_obs_time(self) -> Time:
        time: Time = Time(self.day_obs)
        return time

    @cached_property
    def day_obs_mjd(self) -> int:
        # make mypy happy
        assert isinstance(self.day_obs_time.mjd, float)
        return int(self.day_obs_time.mjd)

    @cached_property
    def day_obs_int(self) -> int:
        # make mypy happy
        assert isinstance(self.day_obs_time.iso, str)
        return int(self.day_obs_time.iso[:10].replace("-", ""))

    @cached_property
    def consdb_visits(self) -> pd.DataFrame:
        # In the schema as of now, all visits are one exposure,
        # and visit_id = exposure_id.
        # To support 2 exposures snaps in the future, this query will
        # need to be rewritten to group by visit_id.
        consdb_visits_query: str = f"""
            SELECT * FROM cdb_{self.instrument}.exposure AS e
            LEFT JOIN cdb_{self.instrument}.visit{self.num_exposures}_quicklook
                AS v{self.num_exposures}q ON e.exposure_id = v{self.num_exposures}q.visit_id
            WHERE e.obs_start_mjd IS NOT NULL
                AND e.s_ra IS NOT NULL
                AND e.s_dec IS NOT NULL
                AND e.sky_rotation IS NOT NULL
                AND ((e.band IS NOT NULL) OR (e.physical_filter IS NOT NULL))
                AND e.day_obs = {self.day_obs_int}
        """
        return query_consdb(consdb_visits_query, self.url)

    @cached_property
    def visit_id(self):
        return self.consdb_visits["visit_id"]

    @cached_property
    def ra(self) -> pd.Series:
        return self.consdb_visits["s_ra"]

    @cached_property
    def decl(self) -> pd.Series:
        return self.consdb_visits["s_dec"]

    @cached_property
    def obs_start_mjd(self) -> pd.Series:
        return self.consdb_visits["obs_start_mjd"]

    @cached_property
    def exp_time(self) -> pd.Series:
        return self.consdb_visits["exp_time"]

    @cached_property
    def sky_rotation(self) -> pd.Series:
        return self.consdb_visits["sky_rotation"]

    @cached_property
    def airmass(self) -> pd.Series:
        consdb_value_dtype = self.consdb_visits["airmass"].dtype

        # Avoid confusing mypy:
        assert isinstance(consdb_value_dtype, np.dtype)

        if np.issubdtype(consdb_value_dtype, np.floating):
            airmass = self.consdb_visits["airmass"]
        else:
            # Calculate airmass using Kirstensen (1998), good to the horizon.
            # Models the atmosphere with a uniform spherical shell with a
            # height 1/470 of the radius of the earth.
            # https://doi.org/10.1002/asna.2123190313
            a_cos_zd = 470 * np.cos(np.radians(self.zd))
            airmass = pd.Series(np.sqrt(a_cos_zd**2 + 941) - a_cos_zd, index=self.consdb_visits.index)

        return airmass

    @cached_property
    def band(self) -> pd.Series:
        return self.consdb_visits["band"]

    @cached_property
    def azimuth(self) -> pd.Series:
        return self.consdb_visits["azimuth_start"]

    @cached_property
    def altitude(self) -> pd.Series:
        return self.consdb_visits["altitude_start"]

    @cached_property
    def zd(self) -> pd.Series:
        return 90 - self.altitude

    @cached_property
    def note(self) -> pd.Series:
        return self.consdb_visits["observation_reason"]

    @cached_property
    def visit_time(self) -> pd.Series:
        return (self.consdb_visits["obs_end_mjd"] - self.consdb_visits["obs_start_mjd"]) * 24 * 60 * 60

    @cached_property
    def cloud(self) -> pd.Series:
        return pd.Series(np.nan, index=self.consdb_visits.index)

    @cached_property
    def slew_distance(self) -> pd.Series:
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
        return Time(self.obs_start_mjd, format="mjd")

    @cached_property
    def observation_start_lst(self) -> pd.Series:
        start_lst = pd.Series(
            self.start_time.sidereal_time("apparent", self.location).degree, index=self.consdb_visits.index
        )
        return start_lst

    @cached_property
    def pseudo_parallactic_angle(self):
        # Following sim_runner
        # Using pseudo_parallactic_angle, see https://smtn-019.lsst.io/v/DM-44258/index.html

        # make mypy happy
        assert isinstance(self.decl.values, np.ndarray)
        assert isinstance(self.ra.values, np.ndarray)
        assert isinstance(self.obs_start_mjd.values, np.ndarray)

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
    def rot_tel_pos(self):
        rot_tel_pos = rotation_converter(telescope=self.telescope)._rotskypos2rottelpos(
            self.sky_rotation, self.pseudo_parallactic_angle
        )
        return rot_tel_pos

    @cached_property
    def parallactic_angle(self):
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
        night: int = 1 + self.day_obs_mjd - int(np.floor(survey_start_mjd()))
        return night

    @cached_property
    def sun_moon_positions(self) -> pd.DataFrame:
        sun_moon = pd.DataFrame(
            self.almanac.get_sun_moon_positions(self.obs_start_mjd.values), index=self.consdb_visits.index
        )

        # Convert to degrees, because everything else is in degrees
        for body in ("sun", "moon"):
            for coord in ("alt", "az", "RA", "dec"):
                column = f"{body}_{coord}"
                sun_moon[column] = np.degrees(sun_moon[column])

        return sun_moon

    @cached_property
    def moon_distance(self) -> pd.Series:
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
        return self.consdb_visits["zero_point_median"] + 2.5 * np.log10(self.gain)

    @cached_property
    def instrumental_zeropoint_e(self) -> pd.Series:
        return self.band.apply(self.instrumental_zeropoint_for_band) + 2.5 * np.log10(self.exp_time)

    @cached_property
    def sky_e_per_pixel(self) -> pd.Series:
        return self.consdb_visits["sky_bg_median"] * self.gain

    @cached_property
    def sky_mag_per_asec(self) -> pd.Series:
        return self.instrumental_zeropoint_e - 2.5 * np.log10(self.sky_e_per_pixel / (self.pixel_scale**2))

    @cache
    def hpix(self, nside) -> pd.Series:
        return pd.Series(hp.ang2pix(nside, self.ra, self.decl, lonlat=True), index=self.consdb_visits.index)

    @cached_property
    def pre_sky_mag_per_asec(self) -> pd.Series:
        sky_model_pre = SkyModelPre(init_load_length=2, load_length=2)
        sky_model_values = pd.Series(np.nan, dtype=float, index=self.consdb_visits.index)

        for id in sky_model_values.index:
            mjd = self.obs_start_mjd[id]
            hpix = self.hpix(sky_model_pre.nside)[id]
            band = self.band[id]
            sky_model_values[id] = sky_model_pre.return_mags(mjd, hpix, [band])[band]

        return sky_model_values

    @cached_property
    def fwhm_eff(self) -> pd.Series:
        return self.consdb_visits["psf_sigma_median"] * GAUSSIAN_FWHM_OVER_SIGMA * self.pixel_scale

    @cached_property
    def fwhm_geom(self) -> pd.Series:
        return SeeingModel.fwhm_eff_to_fwhm_geom(self.fwhm_eff)

    @cached_property
    def fwhm_500(self) -> pd.Series:
        return self.consdb_visits["seeing_zenith_500nm_median"] * GAUSSIAN_FWHM_OVER_SIGMA * self.pixel_scale

    @cached_property
    def opsim(self) -> pd.DataFrame:

        opsim_df = pd.DataFrame(
            {
                "observationId": self.visit_id,
                "fieldRA": self.ra,
                "fieldDec": self.decl,
                "observationStartMJD": self.obs_start_mjd,
                "flush_by_mjd": np.nan,
                "visitExposureTime": self.exp_time,
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
                "fiveSigmaDepth": None,
                "altitude": self.altitude,
                "azimuth": self.azimuth,
                "paraAngle": self.parallactic_angle,
                "psudoParaAngle": self.pseudo_parallactic_angle,
                "cloud": self.cloud,
                "moonAlt": self.sun_moon_positions["moon_alt"],
                "sunAlt": self.sun_moon_positions["sun_alt"],
                "note": self.note,
                "target": None,
                "fieldId": None,
                "proposalId": None,
                "block_id": None,
                "observationStartLST": self.observation_start_lst,
                "rotTelPos": self.rot_tel_pos,
                "rotTelPos_backup": None,
                "moonAz": self.sun_moon_positions["moon_az"],
                "sunAz": self.sun_moon_positions["sun_az"],
                "sunRA": self.sun_moon_positions["sun_RA"],
                "sunDec": self.sun_moon_positions["sun_dec"],
                "moonRA": self.sun_moon_positions["moon_RA"],
                "moonDec": self.sun_moon_positions["moon_dec"],
                "moonDistance": self.moon_distance,
                "solarElong": self.solar_elong,
                "moonPhase": self.sun_moon_positions["moon_phase"],
                "cummTelAz": None,
                "scripted_id": None,
                "start_date": self.consdb_visits["obs_start"],
                "t_eff": self.consdb_visits["eff_time_median"],
                "seq_num": self.consdb_visits["seq_num"],
                "skyBrightnessPre": self.pre_sky_mag_per_asec,
            }
        )

        return opsim_df


class ComcamSimConsDBVisits(ConsDBVisits):

    # Set "constant" properties using cached_property rather
    # than actual attributes, because abstract "members" can only be declared
    # as such using properties, and overriding a property with an actual
    # attribute confuses type checkers.

    @property
    def instrument(self) -> str:
        return "lsstcomcamsim"

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
        return SysEngVals().zp_t[band]

    @property
    def site(self):
        return Site("LSST")

    @cached_property
    def exp_time(self) -> pd.Series:
        consdb_value_dtype = self.consdb_visits["exp_time"].dtype

        # Avoid confusing mypy:
        assert isinstance(consdb_value_dtype, np.dtype)

        if np.issubdtype(consdb_value_dtype, np.number):
            exp_time = self.consdb_visits["exp_time"]
        else:
            exp_time = pd.Series(30, index=self.consdb_visits.index)

        return exp_time

    @cached_property
    def band(self) -> pd.Series:
        band = self.consdb_visits["band"].copy()
        missing_band: pd.Series[bool] = band.isna()
        band[missing_band] = self.consdb_visits.loc[missing_band, "physical_filter"].str.get(0)
        return band


# Different subclasses of ConsDBOpsimConverter are needed for
# different instruments.
# Make a factory function to return the correct one.
def load_consdb_visits(instrument: str = "lsstcomcamsim", *args, **kwargs) -> ConsDBVisits:
    match instrument:
        case "lsstcomcamsim":
            converter: ConsDBVisits = ComcamSimConsDBVisits(*args, **kwargs)
        case _:
            raise NotImplementedError

    return converter
