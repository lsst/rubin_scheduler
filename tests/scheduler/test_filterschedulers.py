import unittest

import numpy as np
from astropy.time import Time

from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.schedulers import ComCamBandSched, DateSwapBandScheduler, SimpleBandSched
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

    def test_DateSwapBandScheduler(self):
        # No specific arguments should be fine
        bandsched = DateSwapBandScheduler()
        # known time with known filters
        tt = Time("2025-08-05T20:00:00")
        conditions = Conditions(nside=8, mjd=tt.mjd)
        self.assertEqual(
            bandsched(conditions),
            [
                "u",
                "r",
                "i",
                "z",
            ],
        )
        # full moon, past the current end of the band scheduler
        tt2 = Time("2025-11-05T20:00:00")
        conditions = Conditions(nside=8, mjd=tt2.mjd)
        conditions.moon_phase_sunset = 100
        self.assertEqual(bandsched(conditions), ["g", "r", "i", "z", "y"])
        # Set values - including override history
        date_swaps = {"2025-08-05": ["u", "g"], "2025-08-06": ["z", "y"]}
        end_date = Time("2025-08-07T12:00:00")
        backup_band_scheduler = SimpleBandSched(illum_limit=50)
        bandsched = DateSwapBandScheduler(
            swap_schedule=date_swaps, end_date=end_date, backup_band_scheduler=backup_band_scheduler
        )
        tt = Time("2025-08-05T20:00:00")
        conditions = Conditions(nside=8, mjd=tt.mjd)
        self.assertEqual(bandsched(conditions), ["u", "g"])
        tt = Time("2025-08-06T20:00:00")
        conditions = Conditions(nside=8, mjd=tt.mjd)
        self.assertEqual(bandsched(conditions), ["z", "y"])
        tt = Time("2025-08-07T20:00:00")
        conditions = Conditions(nside=8, mjd=tt.mjd)
        conditions.moon_phase_sunset = 25
        self.assertEqual(bandsched(conditions), ["u", "g", "r", "i", "z"])
        # And just check that sorting doesn't matter in the input dicts
        date_swaps = {
            "2025-08-06": ["z", "y"],
            "2025-08-05": ["u", "g"],
        }
        end_date = Time("2025-08-07T12:00:00")
        backup_band_scheduler = SimpleBandSched(illum_limit=50)
        bandsched = DateSwapBandScheduler(
            swap_schedule=date_swaps, end_date=end_date, backup_band_scheduler=backup_band_scheduler
        )
        tt = Time("2025-08-05T20:00:00")
        conditions = Conditions(nside=8, mjd=tt.mjd)
        self.assertEqual(bandsched(conditions), ["u", "g"])
        tt = Time("2025-08-06T20:00:00")
        conditions = Conditions(nside=8, mjd=tt.mjd)
        self.assertEqual(bandsched(conditions), ["z", "y"])
        tt = Time("2025-08-07T20:00:00")
        conditions = Conditions(nside=8, mjd=tt.mjd)
        conditions.moon_phase_sunset = 25
        self.assertEqual(bandsched(conditions), ["u", "g", "r", "i", "z"])


if __name__ == "__main__":
    unittest.main()
