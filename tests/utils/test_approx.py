import unittest

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

import rubin_scheduler.utils as utils


def astropy_ra_dec_to_alt_az(ra, dec, mjd, lat_deg, longitude_deg, height=100):
    observing_location = EarthLocation(lat=lat_deg * u.deg, lon=longitude_deg * u.deg, height=height * u.m)
    observing_time = Time(mjd, format="mjd")
    aa = AltAz(location=observing_location, obstime=observing_time)

    coord = SkyCoord(ra * u.deg, dec * u.deg)
    alt_az = coord.transform_to(aa)
    return alt_az.alt.deg, alt_az.az.deg


class ApproxCoordTests(unittest.TestCase):
    """
    Test the fast approximate ra,dec to alt,az transforms
    """

    def test_degrees(self):
        nside = 16
        hpids = np.arange(hp.nside2npix(nside))
        ra, dec = utils.hpid2_ra_dec(nside, hpids)
        mjd = 59852.0
        site = utils.Site("LSST")

        alt1, az1 = astropy_ra_dec_to_alt_az(ra, dec, mjd, site.latitude, site.longitude)
        alt2, az2 = utils.approx_ra_dec2_alt_az(ra, dec, site.latitude, site.longitude, mjd)

        # Check that the fast is similar to the more precice transform
        tol = 2  # Degrees
        tol_mean = 1.0
        separations = utils.angular_separation(az1, alt1, az2, alt2)
        self.assertLess(np.max(separations), tol)
        self.assertLess(np.mean(separations), tol_mean)

        # Check that the fast can nearly round-trip
        ra_back, dec_back = utils.approx_alt_az2_ra_dec(alt2, az2, site.latitude, site.longitude, mjd)
        separations = utils.angular_separation(ra, dec, ra_back, dec_back)
        self.assertLess(np.max(separations), tol)
        self.assertLess(np.mean(separations), tol_mean)

    def test_rad(self):
        nside = 16
        hpids = np.arange(hp.nside2npix(nside))
        ra, dec = utils._hpid2_ra_dec(nside, hpids)
        mjd = 59852.0
        site = utils.Site("LSST")

        alt1, az1 = astropy_ra_dec_to_alt_az(
            np.degrees(ra), np.degrees(dec), mjd, site.latitude, site.longitude
        )
        alt1 = np.radians(alt1)
        az1 = np.radians(az1)

        alt2, az2 = utils._approx_ra_dec2_alt_az(ra, dec, site.latitude_rad, site.longitude_rad, mjd)

        # Check that the fast is similar to the more precice transform
        tol = np.radians(2)
        tol_mean = np.radians(1.0)
        separations = utils._angular_separation(az1, alt1, az2, alt2)

        self.assertLess(np.max(separations), tol)
        self.assertLess(np.mean(separations), tol_mean)

        # Check that the fast can nearly round-trip
        ra_back, dec_back = utils._approx_alt_az2_ra_dec(
            alt2, az2, site.latitude_rad, site.longitude_rad, mjd
        )
        separations = utils._angular_separation(ra, dec, ra_back, dec_back)
        self.assertLess(np.max(separations), tol)
        self.assertLess(np.mean(separations), tol_mean)


if __name__ == "__main__":
    unittest.main()
