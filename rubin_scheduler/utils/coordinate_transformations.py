"""
Coordinate transforms that are more accurate but slower than similar
methods in approx_coord_transforms and may include dependencies on astropy.
"""

__all__ = (
    "alt_az_pa_from_ra_dec",
    "_alt_az_pa_from_ra_dec",
    "spherical_from_cartesian",
    "cartesian_from_spherical",
    "xyz_from_ra_dec",
    "_xyz_from_ra_dec",
    "_ra_dec_from_xyz",
    "ra_dec_from_xyz",
    "xyz_angular_radius",
    "_xyz_angular_radius",
    "rotation_matrix_from_vectors",
    "rot_about_z",
    "rot_about_y",
    "rot_about_x",
    "calc_lmst",
    "calc_lmst_astropy",
    "angular_separation",
    "_angular_separation",
    "haversine",
    "arcsec_from_radians",
    "radians_from_arcsec",
    "arcsec_from_degrees",
    "degrees_from_arcsec",
)

import numbers

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils.iers import conf

from rubin_scheduler.utils.code_utilities import _validate_inputs


def _wrap_hour_angle(ha_rad):
    """wrap hour angle to go between -pi and pi"""

    if np.size(ha_rad) == 1:
        if ha_rad > np.pi:
            ha_rad -= 2.0 * np.pi
    else:
        over = np.where(ha_rad > np.pi)[0]
        ha_rad[over] -= 2.0 * np.pi

    return ha_rad


def alt_az_pa_from_ra_dec(ra, dec, mjd, site_longitude, site_latitude):
    """ """

    alt, az, pa = _alt_az_pa_from_ra_dec(
        np.radians(ra), np.radians(dec), mjd, np.radians(site_longitude), np.radians(site_latitude)
    )

    return np.degrees(alt), np.degrees(az), np.degrees(pa)


def _alt_az_pa_from_ra_dec(ra, dec, mjd, site_longitude, site_latitude):
    """ """

    observing_location = EarthLocation(
        lat=site_latitude * u.rad, lon=site_longitude * u.rad, height=100 * u.m
    )
    observing_time = Time(mjd, format="mjd", location=observing_location)
    aa = AltAz(location=observing_location, obstime=observing_time)
    coord = SkyCoord(ra * u.rad, dec * u.rad)
    altaz = coord.transform_to(aa)

    lmst = observing_time.sidereal_time("mean")

    hour_angle = _wrap_hour_angle(lmst.rad - ra)

    # Position Angle Equation from:
    # http://www.gb.nrao.edu/~rcreager/GBTMetrology/140ft/l0058/
    # gbtmemo52/memo52.html
    # or
    # http://www.gb.nrao.edu/GBT/DA/gbtidl/release2pt9/contrib/
    # contrib/parangle.pro
    pa = np.arctan2(
        np.sin(hour_angle),
        (np.cos(dec) * np.tan(site_latitude) - np.sin(dec) * np.cos(hour_angle)),
    )

    return altaz.alt.rad, altaz.az.rad, pa


def calc_lmst(mjd, longitude_rad):
    """Calculate the LMST for a location
    based on:  https://github.com/jhaupt/Sidereal-Time-Calculator/
    blob/master/SiderealTimeCalculator.py
    which uses:
    http://aa.usno.navy.mil/faq/docs/JD_Formula.php
    http://aa.usno.navy.mil/faq/docs/GAST.php and

    Parameters
    ----------
    mjd : `float`
        is the universal time (ut1) expressed as an MJD.
        This can be a numpy array or a single value.
    long_rad : `float`
        is the longitude in radians (positive east of the prime meridian)
        This can be numpy array or a single value.  If a numpy array,
        should have the same length as mjd.  In that
        case, each long_rad will be applied only to the corresponding mjd.

    Returns
    -------
    lst : `float`
        The local sidereal time in hours

    """
    gmst = 18.697374558 + 24.06570982441908 * (mjd - 51544.5)
    gmst = gmst % 24  # to hours
    longitude = np.degrees(longitude_rad) / 15.0  # Convert longi to hours
    lst = gmst + longitude  # Fraction LST. If negative we want to add 24
    if np.size(lst) == 1:
        if lst < 0:
            lst += 24
    else:
        lst[np.where(lst < 0)] += 24
    return lst


def calc_lmst_astropy(mjd, long_rad):
    """
    calculates local mean sidereal time

    Parameters
    ----------
    mjd : `float`
        is the universal time (ut1) expressed as an MJD.
        This can be a numpy array or a single value.
    long_rad : `float`
        is the longitude in radians (positive east of the prime meridian)
        This can be numpy array or a single value.  If a numpy array,
        should have the same length as mjd.  In that
        case, each long_rad will be applied only to the corresponding mjd.

    Returns
    -------
    lmst : `float`
        is the local mean sidereal time in hours
    """
    observing_location = EarthLocation(lat=0.0, lon=long_rad * u.rad, height=100 * u.m)
    t = Time(mjd, format="mjd", location=observing_location)
    # Ignore astropy if we wander too far into the future
    with conf.set_temp("iers_degraded_accuracy", "warn"):
        lmst = t.sidereal_time("mean").rad

    # convert to hours
    lmst = lmst * 12 / np.pi
    return lmst


def cartesian_from_spherical(longitude, latitude):
    """
    Transforms between spherical and Cartesian coordinates.

    Parameters
    ----------
    longitude : `Unknown`
        is a numpy array or a number in radians
    latitude : `Unknown`
        is a numpy array or number in radians
    a : `Unknown`
        numpy array of the (three-dimensional) cartesian
        coordinates on a unit sphere.

    if inputs are numpy arrays:
    output[i][0] will be the x-coordinate of the ith point
    output[i][1] will be the y-coordinate of the ith point
    output[i][2] will be the z-coordinate of the ith point

    All angles are in radians

    Also, look at xyz_from_ra_dec().
    """
    return _xyz_from_ra_dec(longitude, latitude).transpose()


def spherical_from_cartesian(xyz):
    """
    Transforms between Cartesian and spherical coordinates

    Parameters
    ----------
    xyz : `Unknown`
        is a numpy array of points in 3-D space.
        Each row is a different point.
    returns : `Unknown`
        longitude and latitude

    All angles are in radians

    Also, look at ra_dec_from_xyz().
    """
    if not isinstance(xyz, np.ndarray):
        raise RuntimeError("You need to pass a numpy array to spherical_from_cartesian")

    if len(xyz.shape) > 1:
        return _ra_dec_from_xyz(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    else:
        return _ra_dec_from_xyz(xyz[0], xyz[1], xyz[2])


def xyz_from_ra_dec(ra, dec):
    """
    Utility to convert RA,dec positions in x,y,z space.

    Parameters
    ----------
    ra : float or array
        RA in degrees
    dec : float or array
        Dec in degrees

    Returns
    -------
    x,y,z : floats or arrays
        The position of the given points on the unit sphere.
    """
    return _xyz_from_ra_dec(np.radians(ra), np.radians(dec))


def _xyz_from_ra_dec(ra, dec):
    """
    Utility to convert RA,dec positions in x,y,z space.

    Parameters
    ----------
    ra : float or array
        RA in radians
    dec : float or array
        Dec in radians

    Returns
    -------
    x,y,z : floats or arrays
        The position of the given points on the unit sphere.
    """
    # It is ok to mix floats and numpy arrays.

    cos_dec = np.cos(dec)
    return np.array([np.cos(ra) * cos_dec, np.sin(ra) * cos_dec, np.sin(dec)])


def _ra_dec_from_xyz(x, y, z):
    """
    Utility to convert x,y,z Cartesian coordinates to RA, dec
    positions in space.

    Parameters
    ----------
    x : float or array
        The position on the x-axis of the given points on the unit sphere
    y : float or array
        The position on the y-axis of the given points on the unit sphere
    z : float or array
        The position on the z-axis of the given points on the unit sphere

    Returns
    -------
    ra, dec : floats or arrays
        Ra and dec coordinates in radians.
    """
    rad = np.sqrt(x**2 + y**2 + z**2)
    ra = np.arctan2(y, x)
    dec = np.arcsin(z / rad)

    return ra, dec


def ra_dec_from_xyz(x, y, z):
    """
    Utility to convert x,y,z Cartesian coordinates to RA, dec
    positions in space.

    Parameters
    ----------
    x : float or array
        The position on the x-axis of the given points on the unit sphere
    y : float or array
        The position on the y-axis of the given points on the unit sphere
    z : float or array
        The position on the z-axis of the given points on the unit sphere

    Returns
    -------
    ra, dec : floats or arrays
        Ra and dec coordinates in degrees.
    """

    return np.degrees(_ra_dec_from_xyz(x, y, z))


def xyz_angular_radius(radius=1.75):
    """
    Convert an angular radius into a physical radius for a kdtree search.

    Parameters
    ----------
    radius : float
        Radius in degrees.

    Returns
    -------
    radius : float
    """
    return _xyz_angular_radius(np.radians(radius))


def _xyz_angular_radius(radius):
    """
    Convert an angular radius into a physical radius for a kdtree search.

    Parameters
    ----------
    radius : float
        Radius in radians.

    Returns
    -------
    radius : float
    """
    x0, y0, z0 = (1, 0, 0)
    x1, y1, z1 = _xyz_from_ra_dec(radius, 0)
    result = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    return result


def z_rotation_matrix(theta):
    cc = np.cos(theta)
    ss = np.sin(theta)
    return np.array([[cc, -ss, 0.0], [ss, cc, 0.0], [0.0, 0.0, 1.0]])


def rot_about_z(vec, theta):
    """
    Rotate a Cartesian vector an angle theta about the z axis.
    Theta is in radians.
    Positive theta rotates +x towards +y.
    """
    return np.dot(z_rotation_matrix(theta), vec.transpose()).transpose()


def y_rotation_matrix(theta):
    cc = np.cos(theta)
    ss = np.sin(theta)
    return np.array([[cc, 0.0, ss], [0.0, 1.0, 0.0], [-ss, 0.0, cc]])


def rot_about_y(vec, theta):
    """
    Rotate a Cartesian vector an angle theta about the y axis.
    Theta is in radians.
    Positive theta rotates +x towards -z.
    """
    return np.dot(y_rotation_matrix(theta), vec.transpose()).transpose()


def x_rotation_matrix(theta):
    cc = np.cos(theta)
    ss = np.sin(theta)

    return np.array([[1.0, 0.0, 0.0], [0.0, cc, -ss], [0.0, ss, cc]])


def rot_about_x(vec, theta):
    """
    Rotate a Cartesian vector an angle theta about the x axis.
    Theta is in radians.
    Positive theta rotates +y towards +z.
    """
    return np.dot(x_rotation_matrix(theta), vec.transpose()).transpose()


def rotation_matrix_from_vectors(v1, v2):
    """
    Given two vectors v1,v2 calculate the rotation matrix for v1->v2
    using the axis-angle approach

    Parameters
    ----------
    v1,v2 : `Unknown`
        Cartesian unit vectors (in three dimensions).
    rot : `Unknown`
        is the rotation matrix that rotates from one to the other
    """

    if np.abs(np.sqrt(np.dot(v1, v1)) - 1.0) > 0.01:
        raise RuntimeError("v1 in rotation_matrix_from_vectors is not a unit vector")

    if np.abs(np.sqrt(np.dot(v2, v2)) - 1.0) > 0.01:
        raise RuntimeError("v2 in rotation_matrix_from_vectors is not a unit vector")

    # Calculate the axis of rotation by the cross product of v1 and v2
    cross = np.cross(v1, v2)
    cross = cross / np.sqrt(np.dot(cross, cross))

    # calculate the angle of rotation via dot product
    angle = np.arccos(np.dot(v1, v2))
    sin_dot = np.sin(angle)
    cos_dot = np.cos(angle)

    # calculate the corresponding rotation matrix
    # http://en.wikipedia.org/wiki/Rotation_matrix#
    # Rotation_matrix_from_axis_and_angle
    rot = [
        [
            cos_dot + cross[0] * cross[0] * (1 - cos_dot),
            -cross[2] * sin_dot + (1 - cos_dot) * cross[0] * cross[1],
            cross[1] * sin_dot + (1 - cos_dot) * cross[0] * cross[2],
        ],
        [
            cross[2] * sin_dot + (1 - cos_dot) * cross[0] * cross[1],
            cos_dot + (1 - cos_dot) * cross[1] * cross[1],
            -cross[0] * sin_dot + (1 - cos_dot) * cross[1] * cross[2],
        ],
        [
            -cross[1] * sin_dot + (1 - cos_dot) * cross[0] * cross[2],
            cross[0] * sin_dot + (1 - cos_dot) * cross[1] * cross[2],
            cos_dot + (1 - cos_dot) * (cross[2] * cross[2]),
        ],
    ]

    return rot


def _angular_separation(long1, lat1, long2, lat2):
    """
    Angular separation between two points in radians

    Parameters
    ----------
    long1 is the first longitudinal coordinate in radians

    lat1 is the first latitudinal coordinate in radians

    long2 is the second longitudinal coordinate in radians

    lat2 is the second latitudinal coordinate in radians

    Returns
    -------
    The angular separation between the two points in radians

    Calculated based on the haversine formula
    From http://en.wikipedia.org/wiki/Haversine_formula
    """
    are_arrays_1 = _validate_inputs([long1, lat1], ["long1", "lat1"], "angular_separation")

    are_arrays_2 = _validate_inputs([long2, lat2], ["long2", "lat2"], "angular_separation")

    # The code below is necessary because the call to np.radians() in
    # angular_separation() will automatically convert floats
    # into length 1 numpy arrays, confusing validate_inputs()
    if are_arrays_1 and len(long1) == 1:
        are_arrays_1 = False
        long1 = long1[0]
        lat1 = lat1[0]

    if are_arrays_2 and len(long2) == 1:
        are_arrays_2 = False
        long2 = long2[0]
        lat2 = lat2[0]

    if are_arrays_1 and are_arrays_2:
        if len(long1) != len(long2):
            raise RuntimeError(
                "You need to pass arrays of the same length " "as arguments to angular_separation()"
            )

    t1 = np.sin(lat2 / 2.0 - lat1 / 2.0) ** 2
    t2 = np.cos(lat1) * np.cos(lat2) * np.sin(long2 / 2.0 - long1 / 2.0) ** 2
    _sum = t1 + t2

    if isinstance(_sum, numbers.Number):
        if _sum < 0.0:
            _sum = 0.0
    else:
        _sum = np.where(_sum < 0.0, 0.0, _sum)

    return 2.0 * np.arcsin(np.sqrt(_sum))


def angular_separation(long1, lat1, long2, lat2):
    """
    Angular separation between two points in degrees

    Parameters
    ----------
    long1 is the first longitudinal coordinate in degrees

    lat1 is the first latitudinal coordinate in degrees

    long2 is the second longitudinal coordinate in degrees

    lat2 is the second latitudinal coordinate in degrees

    Returns
    -------
    The angular separation between the two points in degrees
    """
    return np.degrees(
        _angular_separation(np.radians(long1), np.radians(lat1), np.radians(long2), np.radians(lat2))
    )


def haversine(long1, lat1, long2, lat2):
    """
    DEPRECATED; use angular_separation() instead

    Return the angular distance between two points in radians

    Parameters
    ----------
    long1 : `Unknown`
        is the longitude of point 1 in radians
    lat1 : `Unknown`
        is the latitude of point 1 in radians
    long2 : `Unknown`
        is the longitude of point 2 in radians
    lat2 : `Unknown`
        is the latitude of point 2 in radians
    the : `Unknown`
        angular separation between points 1 and 2 in radians
    """
    return _angular_separation(long1, lat1, long2, lat2)


def arcsec_from_radians(value):
    """
    Convert an angle in radians to arcseconds

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return 3600.0 * np.degrees(value)


def radians_from_arcsec(value):
    """
    Convert an angle in arcseconds to radians

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return np.radians(value / 3600.0)


def arcsec_from_degrees(value):
    """
    Convert an angle in degrees to arcseconds

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return 3600.0 * value


def degrees_from_arcsec(value):
    """
    Convert an angle in arcseconds to degrees

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return value / 3600.0
