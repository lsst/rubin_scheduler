import unittest

import numpy as np

from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.schedulers import ComCamBandSched, SimpleBandSched
from rubin_scheduler.utils import SURVEY_START_MJD


class TestBandSchedulers(unittest.TestCase):
    def test_ComCamBandSched(self):
        illum_bins = np.arange(0, 100 + 1, 50)
        band_groups = (("g", "r", "i"), ("i", "z", "y"))
        bandsched = ComCamBandSched(illum_bins=illum_bins, loaded_band_groups=band_groups)
        mjd = SURVEY_START_MJD
        conditions = Conditions(nside=8, mjd=mjd)
        conditions.moon_phase_sunset = 0
        load_bands = bandsched(conditions)
        self.assertTrue(load_bands == ["g", "r", "i"])
        conditions.moon_phase_sunset = 40
        load_bands = bandsched(conditions)
        self.assertTrue(load_bands == ["g", "r", "i"])
        conditions.moon_phase_sunset = 60
        load_bands = bandsched(conditions)
        self.assertTrue(load_bands == ["i", "z", "y"])
        conditions.moon_phase_sunset = 100
        load_bands = bandsched(conditions)
        self.assertTrue(load_bands == ["i", "z", "y"])

    def test_comcambandsched_except(self):
        illum_bins = np.arange(0, 100 + 1, 25)
        band_groups = (("g", "r", "i"), ("i", "z", "y"))
        with self.assertRaises(ValueError):
            ComCamBandSched(illum_bins=illum_bins, loaded_band_groups=band_groups)

    def test_SimpleBandSched(self):
        bandsched = SimpleBandSched(illum_limit=40)
        brightmoon_result = ["g", "r", "i", "z", "y"]
        newmoon_result = ["u", "g", "r", "i", "z"]
        mjd = SURVEY_START_MJD
        conditions = Conditions(nside=8, mjd=mjd)
        conditions.moon_phase_sunset = 0
        load_bands = bandsched(conditions)
        self.assertTrue(load_bands == newmoon_result)
        conditions.moon_phase_sunset = 50
        load_bands = bandsched(conditions)
        self.assertTrue(load_bands == brightmoon_result)


if __name__ == "__main__":
    unittest.main()
