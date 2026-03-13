import unittest

import numpy as np

from rubin_scheduler.utils import mjd2dayobs


class DayobsTests(unittest.TestCase):

    def test_mjd2dayobs(self):

        # Test single value
        mjd = 65320
        day_obs = mjd2dayobs(mjd)
        assert len(day_obs) == 8

        # Test sending in array
        mjd += np.arange(20)
        day_obs = mjd2dayobs(mjd)
        assert len(day_obs) == 20

        # Spot check a few values
        # from night summary
        mjd = 61109.18
        day_obs = mjd2dayobs(mjd)
        assert day_obs == "20260309"

        mjd = 61110.30
        day_obs = mjd2dayobs(mjd)
        assert day_obs == "20260310"


if __name__ == "__main__":
    unittest.main()
