import unittest

import healpy as hp
import numpy as np

from rubin_scheduler.site_models import CloudMap


class TestCloudMap(unittest.TestCase):

    def test_cloudmap(self):
        nside = 32
        n_pix = hp.nside2npix(nside)
        cm = CloudMap()

        n_frames = 10
        rng = np.random.default_rng(seed=42)

        for mjd in np.arange(n_frames):
            cm.add_frame(rng.random(n_pix), mjd)

        map1 = cm.extinction_closest(5)
        assert np.size(map1) == n_pix

        # check that 19 minutes ahead of last frame,
        # still returns closest
        map1 = cm.extinction_closest(mjd + 19 / 60 / 24)
        assert np.size(map1) == n_pix

        # check that going too far from used times
        # returns zero
        map2 = cm.extinction_closest(5000)
        assert map2 == 0

        map3 = cm.extinction_closest(-5000)
        assert map3 == 0

        # Now with uncertainty frames
        cm = CloudMap()
        for mjd in np.arange(n_frames):
            cm.add_frame(rng.random(n_pix), mjd, uncert=rng.random(n_pix))

        # check that 19 minutes ahead of last frame,
        # still returns closest
        map1, uncert1 = cm.extinction_closest(mjd + 19 / 60 / 24, uncert=True)
        assert np.size(map1) == n_pix
        assert np.size(uncert1) == n_pix

        # check that going too far from used times
        # returns zero
        map2, uncert2 = cm.extinction_closest(5000, uncert=True)
        assert map2 == 0
        assert uncert2 == 0


if __name__ == "__main__":
    unittest.main()
