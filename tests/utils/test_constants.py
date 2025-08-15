import unittest

import healpy as hp
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time

import rubin_scheduler.utils as utils


class Constantsests(unittest.TestCase):

    def test_survey_start(self):
        ss_mjd = utils.SURVEY_START_MJD
        ss_time = Time(ss_mjd, format="mjd")

        # Demand that the default start date be during the day
        site = utils.Site("LSST")
        location = EarthLocation(lat=site.latitude, lon=site.longitude, height=site.height)
        sun = get_sun(ss_time)
        aa = AltAz(location=location, obstime=ss_time)
        sun_aa = sun.transform_to(aa)
        assert sun_aa.alt > 0

    def test_nside(self):
        nside = utils.DEFAULT_NSIDE
        npix = hp.nside2npix(nside)
        assert npix > 0


if __name__ == "__main__":
    unittest.main()
