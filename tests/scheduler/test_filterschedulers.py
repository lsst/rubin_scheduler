import unittest

import numpy as np

from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.schedulers import ComCamFilterSched, SimpleFilterSched
from rubin_scheduler.utils import survey_start_mjd


class TestFilterSchedulers(unittest.TestCase):

    def test_ComCamFilterSched(self):
        illum_bins = np.arange(0, 100 + 1, 50)
        filter_groups = (("g", "r", "i"), ("i", "z", "y"))
        filtersched = ComCamFilterSched(illum_bins=illum_bins, loaded_filter_groups=filter_groups)
        mjd = survey_start_mjd()
        conditions = Conditions(nside=8, mjd=mjd)
        conditions.moon_phase_sunset = 0
        load_filters = filtersched(conditions)
        self.assertTrue(load_filters == ["g", "r", "i"])
        conditions.moon_phase_sunset = 40
        load_filters = filtersched(conditions)
        self.assertTrue(load_filters == ["g", "r", "i"])
        conditions.moon_phase_sunset = 60
        load_filters = filtersched(conditions)
        self.assertTrue(load_filters == ["i", "z", "y"])
        conditions.moon_phase_sunset = 100
        load_filters = filtersched(conditions)
        self.assertTrue(load_filters == ["i", "z", "y"])

    def test_comcamfiltersched_except(self):
        illum_bins = np.arange(0, 100 + 1, 25)
        filter_groups = (("g", "r", "i"), ("i", "z", "y"))
        with self.assertRaises(ValueError):
            ComCamFilterSched(illum_bins=illum_bins, loaded_filter_groups=filter_groups)

    def test_SimpleFilterSched(self):
        filtersched = SimpleFilterSched(illum_limit=40)
        brightmoon_result = ["g", "r", "i", "z", "y"]
        newmoon_result = ["u", "g", "r", "i", "z"]
        mjd = survey_start_mjd()
        conditions = Conditions(nside=8, mjd=mjd)
        conditions.moon_phase_sunset = 0
        load_filters = filtersched(conditions)
        self.assertTrue(load_filters == newmoon_result)
        conditions.moon_phase_sunset = 50
        load_filters = filtersched(conditions)
        self.assertTrue(load_filters == brightmoon_result)


if __name__ == "__main__":
    unittest.main()
