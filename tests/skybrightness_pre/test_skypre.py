import unittest
import warnings

import healpy as hp
import numpy as np

import rubin_scheduler.skybrightness_pre as sbp

REMOTE_SKY_URL = (
    "https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/sims_skybrightness_pre/h5_2023_09_12/"
)


class TestSkyPre(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.sm = sbp.SkyModelPre(init_load_length=3, load_length=3)
            mjd = cls.sm.mjds[1] + 4.0 / 60.0 / 24.0
            tmp = cls.sm.return_mags(mjd)
            cls.nside = hp.npix2nside(tmp["r"].size)
            cls.data_present = True
        except ValueError:
            cls.data_present = False
            warnings.warn("Data files not found, skipping tests. Check data/ for instructions to pull data.")

    def test_return_mags(self):
        """
        Test all the ways ReturnMags can be used
        """
        timestep_max = 15.0 / 60.0 / 24.0
        # Check both the healpix and opsim fields
        if self.data_present:
            sms = [self.sm]
            mjds = []
            for mjd in sms[0].mjds[100:102]:
                mjds.append(mjd)
                mjds.append(mjd + 0.0002)

            # Make sure there's an mjd that is between sunrise/set
            # that gets tested
            diff = sms[0].mjds[1:] - sms[0].mjds[0:-1]
            between = np.where(diff >= timestep_max)[0][0]
            mjds.append(sms[0].mjds[between + 1] + timestep_max)

            indxes = [None, [100, 101]]
            bands = [["u", "g", "r", "i", "z", "y"], ["r"]]

            for sm in sms:
                for mjd in mjds:
                    for indx in indxes:
                        for filt in bands:
                            mags = sm.return_mags(mjd, indx=indx, bands=filt)
                            # Check the bands returned are correct
                            self.assertEqual(len(filt), len(list(mags.keys())))
                            self.assertEqual(set(filt), set(mags.keys()))
                            # Check the magnitudes are correct
                            if indx is not None:
                                self.assertEqual(
                                    np.size(mags[list(mags.keys())[0]]),
                                    np.size(indx),
                                )

    def test_bright_sky(self):
        """Test that things behave if request a time with
        a high sun altitude
        """
        # Find an MJD where the sun is high
        diff = np.diff(self.sm.mjds)
        mjd_indx = np.min(np.where(diff == diff.max())[0])
        mjd = np.mean(self.sm.mjds[mjd_indx : mjd_indx + 2])

        with self.assertWarns(Warning):
            mags = self.sm.return_mags(mjd)
            assert np.nanmin(mags["r"]) < 4.0

        # Check that a little over raises the warning and returns
        # closest map
        mjd1 = self.sm.mjds[mjd_indx] - 0.001
        mjd2 = self.sm.mjds[mjd_indx] + 0.001

        # At a valid mjd
        mags1 = self.sm.return_mags(mjd1)

        # MJD slightly beyond where sky was calculated
        with self.assertWarns(Warning):
            mags2 = self.sm.return_mags(mjd2)

        for key in mags1:
            np.allclose(mags1[key], mags2[key])

    def test_various(self):
        """
        Test some various loading things
        """
        # check that the sims_data stuff loads
        sm = sbp.SkyModelPre(init_load_length=3)
        mjd = self.sm.mjds[10] + 0.1
        mags = sm.return_mags(mjd)
        assert mags is not None

    @unittest.skip("Skipping remote sky test (too slow)")
    def test_remote_sky(self):
        sm = sbp.SkyModelPre(init_load_length=3, data_path=REMOTE_SKY_URL)
        min_mjd = np.min(sm.mjds)
        self.assertTrue(40000 < min_mjd < 80000)
        test_mjd = np.ceil(min_mjd) + 0.2
        mags = sm.return_mags(test_mjd)
        for band in mags.keys():
            npix = mags[band].shape[0]
            try:
                hp.npix2nside(npix)
                valid_npix = True
            except ValueError:
                valid_npix = False
            self.assertTrue(valid_npix)


if __name__ == "__main__":
    unittest.main()
