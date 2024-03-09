import unittest

import numpy as np

from rubin_scheduler.utils import survey_start_mjd
from rubin_scheduler.utils import calc_season


class SeasonTest(unittest.TestCase):

    def test_calc_season(self):
        # Test that a single RA value returns a range of season values
        ra = 260.2
        mjd_start = survey_start_mjd()
        mjd = np.arange(mjd_start, mjd_start + 2*365.25)
        # First check with survey_start_mjd within the range of mjds
        # (this also verifies calc_season works with single value ra
        seasons = calc_season(ra, mjd - 100, mjd_start)
        # with the observations spanning the survey start date,
        # the range of 'season' is guaranteed to include 0
        assert len(seasons) == len(mjd)
        assert seasons.min() < 0
        assert seasons.max() > 1
        # Next check with survey_start_mjd long before observations
        seasons = calc_season(ra, mjd+365.25*2, mjd_start)
        assert seasons.min() > 1
        # Next check calc_season works with array of RA values
        ra = np.arange(0, 360, 1)
        seasons = calc_season(ra, mjd, mjd_start)
        assert seasons.shape[0] == len(ra)
        assert seasons.shape[1] == len(mjd)
        # Check that these season values are the same as for a single RA
        i = 20
        one_ra = ra[i]
        one_seasons = calc_season(one_ra, mjd, mjd_start)
        assert np.all(seasons[i] == one_seasons)


if __name__ == "__main__":
    unittest.main()