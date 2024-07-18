__all__ = ("ddf_locations_skycoord", "ddf_locations", "ddf_locations_pre3_5")

import astropy.units as u
from astropy.coordinates import SkyCoord


def ddf_locations_skycoord():
    """Return the DDF locations as a dict of SkyCoord values."""
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
    return ddf


def ddf_locations():
    """Return the DDF locations as a dict in degrees."""
    result = ddf_locations_skycoord()
    for r in result:
        result[r] = (result[r].ra.deg, result[r].dec.deg)

    return result


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
