import copy
import unittest

import numpy as np

import rubin_scheduler.scheduler.detailers as detailers
from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.utils import ObservationArray
from rubin_scheduler.utils import DEFAULT_NSIDE, _ra_dec2_hpid


class TestDetailers(unittest.TestCase):

    def test_basics(self):
        """Test basic detailer functionality"""

        observatory = ModelObservatory()
        conditions = observatory.return_conditions()

        obs_list_orig = []
        dec = np.radians(-20)
        for ra in np.arange(0, 2 * np.pi, np.pi / 4):
            hpid = _ra_dec2_hpid(DEFAULT_NSIDE, ra, dec)
            if np.isfinite(conditions.m5_depth["r"][hpid]):
                obs = ObservationArray()
                obs["filter"] = "r"
                obs["RA"] = ra.copy()
                obs["dec"] = dec
                obs["mjd"] = 59000.0
                obs["exptime"] = 30.0
                obs["scheduler_note"] = "test_note, a"
                obs_list_orig.append(obs)

        for det in detailers.BaseDetailer.__subclasses__():
            obs_list = copy.deepcopy(obs_list_orig)
            live_det = det()
            result = live_det(obs_list, conditions)
            assert len(result) > 0
            # Check that we can add an observation.
            # Should catch anyone who forgot self.features attribute
            live_det.add_observation(obs_list_orig[0])

    def test_random_filter(self):
        obs = ObservationArray(1)
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
