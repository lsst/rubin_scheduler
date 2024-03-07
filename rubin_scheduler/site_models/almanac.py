__all__ = ("Almanac",)

import datetime
import os

import numpy as np
from scipy.interpolate import interp1d

from rubin_scheduler.data import get_data_dir


class Almanac:
    """Class to load and return pre-computed information about the
    LSST site."""

    def __init__(self, mjd_start=None, kind="quadratic"):
        # Load up the sunrise/sunset times
        data_dir = os.path.join(get_data_dir(), "site_models")

        temp = np.load(os.path.join(data_dir, "sunsets.npz"))
        self.sunsets = temp["almanac"].copy()
        temp.close()

        if mjd_start is not None:
            loc = np.searchsorted(self.sunsets["sunset"], mjd_start)
            # Set the start MJD to be night 1.
            self.sunsets["night"] -= self.sunsets["night"][loc - 1]

        # Load the sun and moon positions
        temp = np.load(os.path.join(data_dir, "sun_moon.npz"))
        self.sun_moon = temp["sun_moon_info"].copy()
        temp.close()

        self.interpolators = {
            "sun_alt": interp1d(self.sun_moon["mjd"], self.sun_moon["sun_alt"], kind=kind),
            "sun_az_x": interp1d(self.sun_moon["mjd"], np.cos(self.sun_moon["sun_az"]), kind=kind),
            "sun_az_y": interp1d(self.sun_moon["mjd"], np.sin(self.sun_moon["sun_az"]), kind=kind),
            "sun_RA_x": interp1d(self.sun_moon["mjd"], np.cos(self.sun_moon["sun_RA"]), kind=kind),
            "sun_RA_y": interp1d(self.sun_moon["mjd"], np.sin(self.sun_moon["sun_RA"]), kind=kind),
            "sun_dec": interp1d(self.sun_moon["mjd"], self.sun_moon["sun_dec"], kind=kind),
            "moon_alt": interp1d(self.sun_moon["mjd"], self.sun_moon["moon_alt"], kind=kind),
            "moon_az_x": interp1d(self.sun_moon["mjd"], np.cos(self.sun_moon["moon_az"]), kind=kind),
            "moon_az_y": interp1d(self.sun_moon["mjd"], np.sin(self.sun_moon["moon_az"]), kind=kind),
            "moon_RA_x": interp1d(self.sun_moon["mjd"], np.cos(self.sun_moon["moon_RA"]), kind=kind),
            "moon_RA_y": interp1d(self.sun_moon["mjd"], np.sin(self.sun_moon["moon_RA"]), kind=kind),
            "moon_dec": interp1d(self.sun_moon["mjd"], self.sun_moon["moon_dec"], kind=kind),
            "moon_phase": interp1d(self.sun_moon["mjd"], self.sun_moon["moon_phase"], kind=kind),
        }

        temp = np.load(os.path.join(data_dir, "planet_locations.npz"))
        self.planet_loc = temp["planet_loc"].copy()
        temp.close()

        self.planet_names = ["venus", "mars", "jupiter", "saturn"]
        self.planet_interpolators = {}
        for pn in self.planet_names:
            self.planet_interpolators[pn + "_RA_x"] = interp1d(
                self.planet_loc["mjd"], np.cos(self.planet_loc[pn + "_RA"]), kind=kind
            )
            self.planet_interpolators[pn + "_RA_y"] = interp1d(
                self.planet_loc["mjd"], np.sin(self.planet_loc[pn + "_RA"]), kind=kind
            )
            self.planet_interpolators[pn + "_dec"] = interp1d(
                self.planet_loc["mjd"], self.planet_loc[pn + "_dec"], kind=kind
            )

    def get_planet_positions(self, mjd):
        result = {}
        for pn in self.planet_names:
            result[pn + "_dec"] = self.planet_interpolators[pn + "_dec"](mjd)

            temp_result = np.array(
                [
                    np.arctan2(
                        self.planet_interpolators[pn + "_RA_y"](mjd),
                        self.planet_interpolators[pn + "_RA_x"](mjd),
                    )
                ]
            ).ravel()
            negative_angles = np.where(temp_result < 0.0)[0]
            temp_result[negative_angles] = 2.0 * np.pi + temp_result[negative_angles]
            result[pn + "_RA"] = temp_result
        return result

    def get_sunset_info(self, mjd=None, evening_date=None, longitude=None):
        """Returns a numpy array with mjds for various events
        (sunset, moonrise, sun at -12 degrees alt, etc.).

        Parameters
        ----------
        mjd : `float`
            A UTC MJD that occurs during the desired night.
            Defaults to None.
        evening_date : `str` or `datetime.date`
            The local date of the evening of the night whose index is
            desired, in ISO8601 format (YYYY-MM-DD). Defaults to None.
        longitude : `float` or  `astropy.coordinates.angles.core.Angle`
            If a float, then the value is interpreted as being in radians.
            Defaults to None.

        Returns
        -------
        sunset_info : `numpy.void`
            A numpy object with dtype([
                ('night', '<i8'),
                ('sunset', '<f8'),
                ('sun_n12_setting', '<f8'),
                ('sun_n18_setting', '<f8'),
                ('sun_n18_rising', '<f8'),
                ('sun_n12_rising', '<f8'),
                ('sunrise', '<f8'),
                ('moonrise', '<f8'),
                ('moonset', '<f8')
            ])
        """
        if mjd is not None and evening_date is not None:
            raise ValueError("At most one of mjd and evening_date can be set")

        # Default to now
        if mjd is None and evening_date is None:
            mjd = 40587 + datetime.datetime.now().timestamp() / (24 * 60 * 60)

        if mjd is not None:
            indx = np.searchsorted(self.sunsets["sunset"], mjd, side="right") - 1
        elif evening_date is not None:
            if longitude is None:
                raise ValueError("If evening_date is set, longitude is needed as well")
            indx = self.index_for_local_evening(evening_date, longitude)

        return self.sunsets[indx]

    def mjd_indx(self, mjd):
        indx = np.searchsorted(self.sunsets["sunset"], mjd, side="right") - 1
        return indx

    def index_for_local_evening(self, evening_date, longitude):
        """The index of the night with sunset at a given local date.

        Parameters
        ----------
        evening_date : `str` or `datetime.date`
            The local date of the evening of the night whose index i
            desired, in ISO8601 format (YYYY-MM-DD).
        longitude : `float` or  `astropy.coordinates.angles.core.Angle`
            If a float, then the value is interpreted as being in radians.

        Returns
        -------
        night_index : `int`
            The index of the requested night.
        """
        try:
            longitude = longitude.radian
        except AttributeError:
            pass

        if isinstance(evening_date, str):
            evening_date = datetime.date.fromisoformat(evening_date)

        evening_datetime = datetime.datetime(evening_date.year, evening_date.month, evening_date.day)
        evening_mjd = np.floor(evening_datetime.timestamp() / (24 * 60 * 60) + 40587)

        # Depending on the time of year, the UTC date rollover might not
        # always be on the same side of local sunset. Shift by the
        # longitude to make sure the rollover is always near midnight,
        # far from sunset.
        matching_nights = np.argwhere(
            np.floor(self.sunsets["sunset"] + longitude / (2 * np.pi)) == evening_mjd
        )
        if len(matching_nights) < 1:
            raise ValueError(f"Requested night {evening_date} outside of almanac date range")

        night_index = matching_nights.item()

        return night_index

    def get_sun_moon_positions(self, mjd):
        """
        All angles in Radians. moonPhase between 0 and 100.
        """
        simple_calls = ["sun_alt", "sun_dec", "moon_alt", "moon_dec", "moon_phase"]
        result = {}
        for key in simple_calls:
            result[key] = self.interpolators[key](mjd)

        longitude_calls = ["sun_az", "moon_az", "sun_RA", "moon_RA"]
        for key in longitude_calls:
            # Need to wrap in case sent a scalar
            temp_result = np.array(
                [
                    np.arctan2(
                        self.interpolators[key + "_y"](mjd),
                        self.interpolators[key + "_x"](mjd),
                    )
                ]
            ).ravel()
            negative_angles = np.where(temp_result < 0.0)[0]
            temp_result[negative_angles] = 2.0 * np.pi + temp_result[negative_angles]
            result[key] = temp_result

        return result
