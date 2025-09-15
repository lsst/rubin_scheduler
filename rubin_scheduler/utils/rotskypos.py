__all__ = [
    "rotation_converter",
    "RotationConverter",
    "RotationConverterComCam",
    "RotationConverterAuxtel",
    "pseudo_parallactic_angle",
]

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time


def pseudo_parallactic_angle(
    ra: float | np.ndarray,
    dec: float | np.ndarray,
    mjd: float | np.ndarray,
    lon: float = -70.7494,
    lat: float = -30.2444,
    height: float = 2650.0,
    pressure: float = 750.0,
    temperature: float = 11.5,
    relative_humidity: float = 0.4,
    obswl: float = 1.0,
):
    """Compute the pseudo parallactic angle.

    The (traditional) parallactic angle is the angle zenith - coord - NCP
    where NCP is the true-of-date north celestial pole.  This function instead
    computes zenith - coord - NCP_ICRF where NCP_ICRF is the north celestial
    pole in the International Celestial Reference Frame.
    See: https://smtn-019.lsst.io/v/DM-44258/index.html

    Parameters
    ----------
    ra, dec : `float`
        ICRF coordinates in degrees.
    mjd : `float`
        Modified Julian Date.
    latitude, longitude : float
        Geodetic coordinates of observer in degrees.
    height : `float`
        Height of observer above reference ellipsoid in meters.
    pressure : `float`
        Atmospheric pressure in millibars.
    temperature : `float`
        Atmospheric temperature in degrees Celsius.
    relative_humidity : `float`
    obswl : `float`
        Observation wavelength in microns.

    Returns
    -------
    ppa : float
        The pseudo parallactic angle in degrees.
    alt : float
        Altitude of the observations
    az : float
        Azimuth of the observations
    """
    obstime = Time(mjd, format="mjd", scale="tai")
    location = EarthLocation.from_geodetic(
        lon=lon * u.deg,
        lat=lat * u.deg,
        height=height * u.m,
        ellipsoid="WGS84",  # For concreteness
    )

    coord_kwargs = dict(
        obstime=obstime,
        location=location,
        pressure=pressure * u.mbar,
        temperature=temperature * u.deg_C,
        relative_humidity=relative_humidity,
        obswl=obswl * u.micron,
    )

    coord = SkyCoord(ra * u.deg, dec * u.deg, **coord_kwargs)

    towards_zenith = SkyCoord(
        alt=coord.altaz.alt + 10 * u.arcsec, az=coord.altaz.az, frame=AltAz, **coord_kwargs
    )

    towards_north = SkyCoord(ra=coord.icrs.ra, dec=coord.icrs.dec + 10 * u.arcsec, **coord_kwargs)

    ppa = coord.position_angle(towards_zenith) - coord.position_angle(towards_north)
    return ppa.wrap_at(180 * u.deg).deg, coord.altaz.alt.deg, coord.altaz.az.deg


def rotation_converter(telescope="rubin"):
    """Return the correct RotationConverter object."""
    if telescope.lower() == "rubin":
        return RotationConverter()
    elif telescope.lower() == "auxtel":
        return RotationConverterAuxtel()
    elif telescope.lower() == "comcam":
        return RotationConverterComCam()
    else:
        raise ValueError("Unknown telescope name")


def _wrap_180(in_angle):
    """Convert angle(s) to run from -180 to 180

    Parameters
    ----------
    in_angle : `float`
        Input angle in radians.
    """
    # angle = np.atan2(np.sin(in_angle), np.cos(in_angle))
    # would be simpler, but maybe slower?
    angle = in_angle % (2.0 * np.pi)
    if np.isscalar(angle):
        if angle > np.pi:
            result = angle - 2.0 * np.pi
            return result
        else:
            return angle
    else:
        indx = np.where(angle > np.pi)[0]
        angle[indx] = angle[indx] - 2.0 * np.pi
        return angle


class RotationConverter(object):
    """Class to convert between rotTelPos and rotSkyPos for LSSTcam."""

    def rottelpos2rotskypos(self, rottelpos_in, pa):
        """convert rotTelPos to rotSkyPos

        Parameters
        ----------
        rottelpos_in : `float`
            RotTelPos value in degrees
        pa : `float`
            Parallactic angle in degrees.
        """
        return np.degrees(self._rottelpos2rotskypos(np.radians(rottelpos_in), np.radians(pa)))

    def rotskypos2rottelpos(self, rotskypos_in, pa):
        """convert rotSkyPos to rotTelPos

        Parameters
        ----------
        rotskypos_in : `float`
            RotSkyPos value in degrees
        pa : `float`
            Parallactic angle in degrees.
        """
        result = self._rotskypos2rottelpos(np.radians(rotskypos_in), np.radians(pa))
        return np.degrees(result)

    def _rottelpos2rotskypos(self, rottelpos_in, pa):
        result = (pa - rottelpos_in - np.pi / 2) % (2.0 * np.pi)
        return result

    def _rotskypos2rottelpos(self, rotskypos_in, pa):
        result = (pa - rotskypos_in - np.pi / 2) % (2.0 * np.pi)
        # Enforce rotTelPos between -pi and pi
        return _wrap_180(result)


class RotationConverterComCam(RotationConverter):
    """Class to convert between rotTelPos and rotSkyPos for ComCam."""

    def _rottelpos2rotskypos(self, rottelpos_in, pa):
        result = (pa - rottelpos_in) % (2.0 * np.pi)
        return result

    def _rotskypos2rottelpos(self, rotskypos_in, pa):
        result = (pa - rotskypos_in) % (2.0 * np.pi)
        # Enforce rotTelPos between -pi and pi
        return _wrap_180(result)


class RotationConverterAuxtel(RotationConverter):
    """Use a different relation for rotation angles on AuxTel"""

    def _rottelpos2rotskypos(self, rottelpos_in, pa):
        return (rottelpos_in - pa) % (2.0 * np.pi)

    def _rotskypos2rottelpos(self, rotskypos_in, pa):
        result = (rotskypos_in + pa) % (2.0 * np.pi)
        # Enforce rotTelPos between -pi and pi
        return _wrap_180(result)
