import unittest

import healpy as hp
import numpy as np

from rubin_scheduler.skybrightness_pre import SkyModelPre

OUT_OF_RANGE_MJD = 99999
# 2132-08-31


class TestSkyOutOfRange(unittest.TestCase):
    def test_init_date_out_of_range(self):
        def make_oor_sky_model():
            SkyModelPre(init_load_length=3, load_length=3, mjd0=OUT_OF_RANGE_MJD)

        self.assertRaises(ValueError, make_oor_sky_model)

    def test_request_date_out_of_range(self):
        sky_model = SkyModelPre(init_load_length=3, load_length=3)

        def request_oor_sky():
            mags = sky_model.return_mags(OUT_OF_RANGE_MJD)
            return mags

        self.assertRaises(ValueError, request_oor_sky)

    def test_recover_from_oor_request(self):
        sky_model = SkyModelPre(init_load_length=3, load_length=3)
        sample_mjd_in_range = sky_model.mjds[1]

        try:
            mags = sky_model.return_mags(OUT_OF_RANGE_MJD)
        except ValueError:
            pass

        # Is the sky_model instance still usable, even after
        # we've asked if for an out of range date?
        mags = sky_model.return_mags(sample_mjd_in_range)
        for band in "ugrizy":
            self.assertIsInstance(mags[band], np.ndarray)

            # This will raise an exception if the array size isn't
            # valid healpy size
            hp.npix2nside(mags[band].shape[0])

            # Are the values the array contains reasonable?
            self.assertGreater(np.nanmin(mags[band]), 10)
            self.assertLess(np.nanmax(mags[band]), 30)
