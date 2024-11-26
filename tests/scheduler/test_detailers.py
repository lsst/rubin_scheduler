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

        det_list = [
            detailers.VaryExptDetailer,
            detailers.ZeroRotDetailer,
            detailers.Comcam90rotDetailer,
            detailers.Rottep2RotspDesiredDetailer,
            detailers.CloseAltDetailer,
            detailers.TakeAsPairsDetailer,
            detailers.TwilightTripleDetailer,
            detailers.FlushForSchedDetailer,
            detailers.FilterNexp,
            detailers.FixedSkyAngleDetailer,
            detailers.ParallacticRotationDetailer,
            detailers.FlushByDetailer,
            detailers.RandomFilterDetailer,
            detailers.TrackingInfoDetailer,
            detailers.AltAz2RaDecDetailer,
            detailers.DitherDetailer,
            detailers.EuclidDitherDetailer,
            detailers.CameraRotDetailer,
            detailers.CameraSmallRotPerObservationListDetailer,
        ]

        for det in det_list:
            obs_list = copy.deepcopy(obs_list_orig)
            live_det = det()
            result = live_det(obs_list, conditions)
            assert len(result) > 0

    def test_start_field(self):

        observatory = ModelObservatory()
        conditions = observatory.return_conditions()

        scheduler_note = "prepended"

        obs_to_prepend = [ObservationArray(n=1)] * 3
        for i, obs in enumerate(obs_to_prepend):
            obs["RA"] = np.radians(20)
            obs["filter"] = "r"
            obs["scheduler_note"] = scheduler_note

        obs_reg = [ObservationArray(n=1)]
        det = detailers.StartFieldSequenceDetailer(obs_to_prepend, scheduler_note=scheduler_note)
        obs_out = det(obs_reg, conditions)

        assert len(obs_out) == 4

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
