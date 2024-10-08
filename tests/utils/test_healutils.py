import unittest

import healpy as hp
import numpy as np

import rubin_scheduler.utils as utils


class TestHealUtils(unittest.TestCase):
    def test_ra_decs_rad(self):
        """
        Test that the Ra Dec conversions round-trip
        """

        nside = 64
        hpids = np.arange(hp.nside2npix(nside))
        ra, dec = utils._hpid2_ra_dec(nside, hpids)

        hpids_return = utils._ra_dec2_hpid(nside, ra, dec)

        np.testing.assert_array_equal(hpids, hpids_return)

    def test_ra_decs_deg(self):
        """
        Test that the Ra Dec conversions round-trip
        """

        nside = 64
        hpids = np.arange(hp.nside2npix(nside))
        ra, dec = utils.hpid2_ra_dec(nside, hpids)

        hpids_return = utils.ra_dec2_hpid(nside, ra, dec)

        np.testing.assert_array_equal(hpids, hpids_return)

    def test_bin_rad(self):
        """
        Test that healbin returns correct values and valid healpy maps.
        """

        ra = np.zeros(3)
        dec = np.zeros(3)
        values = ra * 0.0 + 1.0

        nside = 128
        hpid = utils._ra_dec2_hpid(nside, ra[0], dec[0])

        map1 = utils._healbin(ra, dec, values, nside=nside)
        self.assertEqual(map1[hpid], 1.0)
        self.assertEqual(hp.maptype(map1), 0)
        map2 = utils._healbin(ra, dec, values, nside=nside, reduce_func=np.sum)
        self.assertEqual(map2[hpid], 3.0)
        self.assertEqual(hp.maptype(map2), 0)
        map3 = utils._healbin(ra, dec, values, nside=nside, reduce_func=np.std)
        self.assertEqual(map3[hpid], 0.0)
        self.assertEqual(hp.maptype(map3), 0)

    def test_bin_deg(self):
        """
        Test that healbin returns correct values and valid healpy maps.
        """

        ra = np.zeros(3)
        dec = np.zeros(3)
        values = ra * 0.0 + 1.0

        nside = 128
        hpid = utils.ra_dec2_hpid(nside, ra[0], dec[0])

        map1 = utils.healbin(ra, dec, values, nside=nside)
        self.assertEqual(map1[hpid], 1.0)
        self.assertEqual(hp.maptype(map1), 0)
        map2 = utils.healbin(ra, dec, values, nside=nside, reduce_func=np.sum)
        self.assertEqual(map2[hpid], 3.0)
        self.assertEqual(hp.maptype(map2), 0)
        map3 = utils.healbin(ra, dec, values, nside=nside, reduce_func=np.std)
        self.assertEqual(map3[hpid], 0.0)
        self.assertEqual(hp.maptype(map3), 0)

    def test_mask_grow(self):
        """Test we can grow a healpix mask map"""

        nside = 32
        nan_indx = tuple([0, 100])
        scale = hp.nside2resol(nside)

        to_mask = utils._hp_grow_mask(nside, nan_indx, grow_size=scale * 2)

        # That should have made some things mask
        assert 100 > np.size(to_mask) > 5

        # Test another nside
        nside = 128
        nan_indx = tuple([0, 100])
        scale = hp.nside2resol(nside)
        to_mask = utils._hp_grow_mask(nside, nan_indx, grow_size=scale * 2)

        # That should have made some things mask
        assert 100 > np.size(to_mask) > 5

        # Do a silly loop check to make sure things got masked properly
        # Need to turn off the machine precision kwarg
        nside = 128
        nan_indx = tuple([0, 100])
        for scale in np.radians(
            [
                0.1,
                1.0,
                10.0,
                30,
            ]
        ):
            to_mask = utils._hp_grow_mask(nside, nan_indx, grow_size=scale, scale=None)

            ra, dec = utils._hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))
            to_mask_set = set(to_mask)
            all_close = []
            for indx in nan_indx:
                distances = utils._angular_separation(ra, dec, ra[indx], dec[indx])
                close = np.where(distances <= scale)[0]
                all_close.extend(close)
                assert set(close).issubset(to_mask_set)
            all_close = np.unique(all_close)
            # Make sure we aren't including any extra pixels
            assert np.size(all_close) == np.size(to_mask)

        # Check that 0 distance doesn't mask anything new
        to_mask = utils._hp_grow_mask(nside, nan_indx, grow_size=0)
        assert np.size(to_mask) == np.size(nan_indx)
        # Check scale can be wiped out.
        to_mask = utils._hp_grow_mask(nside, nan_indx, grow_size=0, scale=None)
        assert np.size(to_mask) == np.size(nan_indx)


if __name__ == "__main__":
    unittest.main()
