import unittest

import rubin_scheduler.scheduler.detailers as detailers
from rubin_scheduler.scheduler.utils import empty_observation
from rubin_scheduler.scheduler.features import Conditions


class TestDetailers(unittest.TestCase):

    def test_random_filter(self):
        obs = empty_observation(1)
        obs["filter"] = "r"

        det = detailers.RandomFilterDetailer(filters="iz")

        conditions = Conditions()
        conditions.night = 3
        conditions.mounted_filters = ["i", "z"]

        out_obs = det(obs, conditions)
        assert (out_obs["filter"] == "i") | (out_obs["filter"] == "z")

        # Check that we fall back properly
        conditions.mounted_filters = ["r", "g", "u", "y"]
        det = detailers.RandomFilterDetailer(filters="iz", fallback_order="y")
        out_obs = det(obs, conditions)

        assert out_obs["filter"] == "y"


if __name__ == "__main__":
    unittest.main()