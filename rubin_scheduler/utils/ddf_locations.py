__all__ = ("ddf_locations", "ddf_locations_pre3_5", "special_locations", "ddf_locations_skycoord")

import warnings

import astropy.units as u
from astropy.coordinates import SkyCoord


def special_locations(skycoords=False):
    """Special locations.

    Parameters
    ----------
    skycoords : `bool`
        Return locations as astropy.SkyCoords. If False, returns
        as a tuple of floats. Default False.
    """

    # Roman bulge field location
    roman_ra_deg = 268.708
    roman_dec_deg = -28.975

    result = {}
    if skycoords:
        result["Roman_bulge_location"] = SkyCoord(roman_ra_deg * u.deg, roman_dec_deg * u.deg, frame="icrs")
        result.update(ddf_locations(skycoords=True))
    else:
        result["Roman_bulge_location"] = (roman_ra_deg, roman_dec_deg)
        result.update(ddf_locations())

    return result


def ddf_locations_skycoord():
    warnings.warn("ddf_locations_skycoord deprecated, use ddf_locations(skycoords=True)", FutureWarning)
    return ddf_locations(skycoords=True)


def ddf_locations(skycoords=False):
    """Return the DDF locations as a dict in degrees."""
    # The DDF locations here are from Neil Brandt's white paper
    # submitted in response to the 2018 Call for White Papers on observing
    # strategy
    # Document-30468 -- AGN-DDF-WP-02.pdf
    # The locations are chosen based on existing multi-wavelength
    # coverage, plus an offset to avoid the bright star Mira near XMM-LSS

    ddf = {}
    ddf["ELAISS1"] = SkyCoord("00:37:48 −44:01:30", unit=(u.hourangle, u.deg), frame="icrs")
    ddf["XMM_LSS"] = SkyCoord("02:22:18  −04:49:00", unit=(u.hourangle, u.deg), frame="icrs")
    ddf["ECDFS"] = SkyCoord("03:31:55  −28:07:00", unit=(u.hourangle, u.deg), frame="icrs")
    ddf["COSMOS"] = SkyCoord("10:00:26  +02:14:01", unit=(u.hourangle, u.deg), frame="icrs")
    ddf["EDFS_a"] = SkyCoord(ra=58.90 * u.deg, dec=-49.32 * u.deg, frame="icrs")
    ddf["EDFS_b"] = SkyCoord(ra=63.60 * u.deg, dec=-47.60 * u.deg, frame="icrs")

    if not skycoords:
        # replace SkyCoord with ra/deg tuple in degrees
        ddf = dict([(key, (ddf[key].ra.deg, ddf[key].dec.deg)) for key in ddf])

    return ddf


def ddf_locations_pre3_5():
    """Return the DDF locations used for v1 to v3.4 simulations."""
    ddf = {}
    ddf["ELAISS1"] = (9.45, -44.0)
    ddf["XMM_LSS"] = (35.708333, -4 - 45 / 60.0)
    ddf["ECDFS"] = (53.125, -28.0 - 6 / 60.0)
    ddf["COSMOS"] = (150.1, 2.0 + 10.0 / 60.0 + 55 / 3600.0)
    ddf["EDFS_a"] = (58.90, -49.315)
    ddf["EDFS_b"] = (63.6, -47.60)
    return ddf
