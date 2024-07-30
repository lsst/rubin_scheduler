import importlib.util
import urllib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

# import rubin_sim
from astropy.coordinates import angular_separation
from astropy.coordinates.earth import EarthLocation
from astropy.time import Time

from rubin_scheduler.site_models import Almanac
from rubin_scheduler.utils import (
    Site,
    _angular_separation,
    _approx_altaz2pa,
    pseudo_parallactic_angle,
    rotation_converter,
    survey_start_mjd,
)

SUPPORT_HTTP = (
    importlib.util.find_spec("lsst")
    and importlib.util.find_spec("lsst.summit")
    and importlib.util.find_spec("lsst.summit.utils")
)
if SUPPORT_HTTP:
    from lsst.summit.utils import ConsDbClient

SUPPORT_POSTGRESQL = importlib.util.find_spec("sqlalchemy") and importlib.util.find_spec("psycopg2")
if SUPPORT_POSTGRESQL:
    import sqlalchemy


def query_consdb(query: str, url: str = "postgresql://usdf@usdf-summitdb.slac.stanford.edu:5432/exposurelog"):
    url_scheme: str = urllib.parse.urlparse(url).scheme
    match url_scheme:
        case "postgresql":
            if not SUPPORT_POSTGRESQL:
                raise RuntimeError("Optional dependencies required for postgresql access not installed")

            connection = sqlalchemy.create_engine(url)
            query_results: pd.DataFrame = pd.read_sql(query, connection)
        case "http" | "https":
            if not SUPPORT_HTTP:
                raise RuntimeError("Optional dependencies required for ConsDB access not installed")

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

    @classmethod
    @property
    @abstractmethod
    def instrument(cls) -> str:
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def num_exposures(cls) -> int:
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def telescope(cls) -> str:
        raise NotImplementedError

    @cached_property
    def almanac(self):
        return Almanac()

    @cached_property
    @abstractmethod
    def site(self):
        raise NotImplementedError

    @cached_property
    def location(self):
        return EarthLocation(lat=self.site.latitude, lon=self.site.longitude, height=self.site.height)

    @cached_property
    def day_obs_mjd(self) -> int:
        return int(Time(self.day_obs).mjd)

    @cached_property
    def day_obs_int(self) -> int:
        return int(Time(self.day_obs_mjd, format="mjd").iso[:10].replace("-", ""))

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
        return query_consdb(consdb_visits_query, self.url)  # .set_index("visit_id")

    @cached_property
    def visit_id(self):
        return self.consdb_visits["visit_id"]
        #        return self.consdb_visits.index.to_series()

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
        return self.consdb_visits["airmass"]

    @cached_property
    def inferred_band(self) -> pd.Series:
        band = self.consdb_visits["band"].copy()
        missing_band: pd.Series[bool] = band.isna()
        band[missing_band] = self.consdb_visits.loc[missing_band, "physical_filter"].str.get(0)
        return band

    @cached_property
    def azimuth(self) -> pd.Series:
        return self.consdb_visits["azimuth_start"]

    @cached_property
    def altitude(self) -> pd.Series:
        return self.consdb_visits["altitude_start"]

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
        moon_distance_rad = _angular_separation(
            np.radians(self.ra.values),
            np.radians(self.decl.values),
            np.radians(self.sun_moon_positions["moon_RA"].values),
            np.radians(self.sun_moon_positions["sun_dec"].values),
        )
        return pd.Series(np.degrees(moon_distance_rad), index=self.consdb_visits.index)

    @cached_property
    def solar_elong(self) -> pd.Series:
        solar_elong_rad = _angular_separation(
            np.radians(self.ra.values),
            np.radians(self.decl.values),
            np.radians(self.sun_moon_positions["sun_RA"].values),
            np.radians(self.sun_moon_positions["sun_dec"].values),
        )
        return pd.Series(np.degrees(solar_elong_rad), index=self.consdb_visits.index)

    @property
    def visits(self) -> pd.DataFrame:

        visits: pd.DataFrame = pd.DataFrame(
            {
                v: self.consdb_visits[k]
                for k, v in self.exposure_opsimdb_map.items()
                if k in self.consdb_visits
            }
        )

        self._replace_missing_filters(visits, self.consdb_visits)

        if "visitTime" not in visits.columns:
            visits["visitTime"] = self._compute_visit_times(self.consdb_visits)

        self._replace_missing_visit_exposure_times(visits, self.consdb_visits)

        if "slewDistance" not in visits.columns:
            visits["slewDistance"] = self._compute_slew_distance(visits)

        start_times = Time(visits["observationStartMJD"], format="mjd")

        if "observationStartLST" not in visits.columns:
            visits["observationStartLST"] = self._compute_observation_start_lst(start_times)

        if "cloud" not in visits.columns:
            visits["cloud"] = self._compute_cloud(visits)

        if "num_exposures" not in visits.columns:
            visits["numExposures"] = self.num_exposures

        # Locations of sun and moon
        coord_map = {"alt": "Alt", "az": "Az", "RA": "RA", "dec": "Dec"}
        sun_moon_positions = self.almanac.get_sun_moon_positions(visits["observationStartMJD"])
        for body in "moon", "sun":
            for almanac_coord in coord_map:
                opsim_coord = coord_map[almanac_coord]
                opsim_key = f"{body}{opsim_coord}"
                almanac_key = f"{body}_{almanac_coord}"
                if opsim_key not in visits.columns:
                    visits[opsim_key] = np.degrees(sun_moon_positions[almanac_key])

        if "moonPhase" not in visits.columns:
            visits["moonPhase"] = sun_moon_positions["moon_phase"]

        if "moonDistance" not in visits.columns:
            visits["moonDistance"] = self._compute_moon_distance(visits)

        if "solarElong" not in visits.columns:
            visits["solarElong"] = self._compute_solar_elong(visits)

        if "psudoParaAngle" not in visits.columns:
            visits["psudoParaAngle"] = self._compute_psudo_pa(visits)

        if "paraAngle" not in visits.columns:
            visits["paraAngle"] = self._compute_pa(visits)

        if "retTolPos" not in visits.columns:
            visits["rotTelPos"] = self._compute_rot_tel_pos(visits)

        if "night" not in visits.columns:
            visits["night"] = self._compute_night(self.day_obs_mjd)

        if self.stackers is not None:
            visit_records: np.recarray = visits.to_records()
            for stacker in self.stackers:
                visit_records = stacker.run(visit_records)
            visits = pd.DataFrame(visit_records)

        return visits

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
                "filter": self.inferred_band,
                "rotSkyPos": self.sky_rotation,
                "rotSkyPos_desired": np.nan,
                "numExposures": self.num_exposures,
                "airmass": self.airmass,
                "seeingFwhm500": None,
                "seeingFwhmEff": None,
                "seeingFwhmGeom": None,
                "skyBrightness": None,
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
            }
        )

        return opsim_df


class ComcamSimConsDBVisits(ConsDBVisits):
    instrument: str = "lsstcomcamsim"
    num_exposures: int = 1
    telescope: str = "rubin"

    @cached_property
    @abstractmethod
    def site(self):
        return Site("LSST")


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
