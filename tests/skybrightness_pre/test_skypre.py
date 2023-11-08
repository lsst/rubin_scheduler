import unittest
import warnings

import healpy as hp
import numpy as np

import rubin_scheduler.skybrightness_pre as sbp


class TestSkyPre(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.sm = sbp.SkyModelPre(init_load_length=3, load_length=3)
            mjd = cls.sm.mjds[1] + 4.0 / 60.0 / 24.0
            tmp = cls.sm.return_mags(mjd)
            cls.nside = hp.npix2nside(tmp["r"].size)
            cls.data_present = True
        except:
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

            # Make sure there's an mjd that is between sunrise/set that gets tested
            diff = sms[0].mjds[1:] - sms[0].mjds[0:-1]
            between = np.where(diff >= timestep_max)[0][0]
            mjds.append(sms[0].mjds[between + 1] + timestep_max)

            indxes = [None, [100, 101]]
            filters = [["u", "g", "r", "i", "z", "y"], ["r"]]

            for sm in sms:
                for mjd in mjds:
                    for indx in indxes:
                        for filt in filters:
                            mags = sm.return_mags(mjd, indx=indx, filters=filt)
                            # Check the filters returned are correct
                            self.assertEqual(len(filt), len(list(mags.keys())))
                            self.assertEqual(set(filt), set(mags.keys()))
                            # Check the magnitudes are correct
                            if indx is not None:
                                self.assertEqual(
                                    np.size(mags[list(mags.keys())[0]]),
                                    np.size(indx),
                                )

    def test_various(self):
        """
        Test some various loading things
        """
        # check that the sims_data stuff loads
        sm = sbp.SkyModelPre(init_load_length=3)
        mjd = self.sm.mjds[10] + 0.1
        mags = sm.return_mags(mjd)
        assert mags is not None


if __name__ == "__main__":
    unittest.main()
