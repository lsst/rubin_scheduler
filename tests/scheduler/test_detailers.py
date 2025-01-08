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

        dec = np.radians(-20)
        ra = np.arange(0, 2 * np.pi, np.pi / 4)

        obs_array = ObservationArray(n=ra.size)
        obs_array["band"] = "r"
        obs_array["RA"] = ra
        obs_array["dec"] = dec
        obs_array["mjd"] = 59000.0
        obs_array["exptime"] = 30.0
        obs_array["scheduler_note"] = "test_note, a"

        hpid = _ra_dec2_hpid(DEFAULT_NSIDE, ra, dec)

        indx = np.isfinite(conditions.m5_depth["r"][hpid])
        obs_array = obs_array[indx]

        det_list = [
            detailers.VaryExptDetailer,
            detailers.ZeroRotDetailer,
            detailers.Comcam90rotDetailer,
            detailers.Rottep2RotspDesiredDetailer,
            detailers.CloseAltDetailer,
            detailers.TakeAsPairsDetailer,
            detailers.TwilightTripleDetailer,
            detailers.FlushForSchedDetailer,
            detailers.BandNexp,
            detailers.FixedSkyAngleDetailer,
            detailers.ParallacticRotationDetailer,
            detailers.FlushByDetailer,
            detailers.RandomBandDetailer,
            detailers.TrackingInfoDetailer,
            detailers.AltAz2RaDecDetailer,
            detailers.DitherDetailer,
            detailers.EuclidDitherDetailer,
            detailers.CameraRotDetailer,
            detailers.CameraSmallRotPerObservationListDetailer,
            detailers.BandToFilterDetailer,
        ]

        for det in det_list:
            obs = copy.deepcopy(obs_array)
            live_det = det()
            result = live_det(obs, conditions)
            assert len(result) > 0

    def test_start_field(self):
        observatory = ModelObservatory()
        conditions = observatory.return_conditions()

        scheduler_note = "prepended"

        obs_to_prepend = ObservationArray(n=3)
        obs_to_prepend["RA"] = np.radians(20)
        obs_to_prepend["band"] = "r"
        obs_to_prepend["scheduler_note"] = scheduler_note

        obs_reg = ObservationArray(n=1)
        det = detailers.StartFieldSequenceDetailer(obs_to_prepend, scheduler_note=scheduler_note)
        obs_out = det(obs_reg, conditions)

        assert len(obs_out) == 4

    def test_random_band(self):
        obs = ObservationArray(1)
        obs["band"] = "r"

        det = detailers.RandomBandDetailer(bands="iz")

        conditions = Conditions()
        conditions.night = 3
        conditions.mounted_bands = ["i", "z"]

        out_obs = det(obs, conditions)
        assert (out_obs["band"] == "i") | (out_obs["filter"] == "z")

        # Check that we fall back properly
        conditions.mounted_bands = ["r", "g", "u", "y"]
        det = detailers.RandomBandDetailer(bands="iz", fallback_order="y")
        out_obs = det(obs, conditions)

        assert out_obs["band"] == "y"


if __name__ == "__main__":
    unittest.main()
